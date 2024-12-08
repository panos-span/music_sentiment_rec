# A full LSTM implementation is provided
# You can use this or the one you implemented in the second lab

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Subset
from dataset import SpectrogramDataset, torch_train_val_split, CLASS_MAPPING
import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from plot_confusion_matrix import plot_confusion_matrix
from pathlib import Path


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_default_dtype(torch.float32)  # New modern approach
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Detect if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Directory to save checkpoints
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

# Hyperparameters
lr = 1e-4
weight_decay = 1e-4
num_epochs = 200
batch_size = 32
dropout = 0.4
rnn_size = 64
num_layers = 2
bidirectional = False
base_path = Path('./data')

class EarlyStopping:
    def __init__(self, patience, mode="min", base=None):
        self.best = base
        self.patience = patience
        self.patience_left = patience
        self.mode = mode
        
    def stop(self, value: float) -> bool:
        if self.has_improved(value):
            self.patience_left = self.patience
            self.best = value
        else:
            self.patience_left -= 1
        return self.patience_left <= 0
    
    def is_best(self, value: float) -> bool:
        return self.has_improved(value)
    
    def has_improved(self, value: float) -> bool:
        return (self.best is None) or (
            value < self.best if self.mode == "min" else value > self.best
        )


class PadPackedSequence(nn.Module):
    def __init__(self):
        """Wrap sequence padding in nn.Module
        Args:
            batch_first (bool, optional): Use batch first representation. Defaults to True.
        """
        super(PadPackedSequence, self).__init__()
        self.batch_first = True
        self.max_length = None

    def forward(self, x):
        """Convert packed sequence to padded sequence
        Args:
            x (torch.nn.utils.rnn.PackedSequence): Packed sequence
        Returns:
            torch.Tensor: Padded sequence
        """
        out, lengths = pad_packed_sequence(
            x, batch_first=self.batch_first, total_length=self.max_length  # type: ignore
        )
        lengths = lengths.to(out.device)
        return out, lengths  # type: ignore


class PackSequence(nn.Module):
    def __init__(self):
        """Wrap sequence packing in nn.Module
        Args:
            batch_first (bool, optional): Use batch first representation. Defaults to True.
        """
        super(PackSequence, self).__init__()
        self.batch_first = True

    def forward(self, x, lengths):
        """Pack a padded sequence and sort lengths
        Args:
            x (torch.Tensor): Padded tensor
            lengths (torch.Tensor): Original lengths befor padding
        Returns:
            Tuple[torch.nn.utils.rnn.PackedSequence, torch.Tensor]: (packed sequence, sorted lengths)
        """
        lengths = lengths.to("cpu")
        out = pack_padded_sequence(
            x, lengths, batch_first=self.batch_first, enforce_sorted=False
        )

        return out

class LSTMBackbone(nn.Module):
    def __init__(
        self,
        input_dim,
        rnn_size=rnn_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
    ):
        super(LSTMBackbone, self).__init__()
        self.batch_first = True
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size

        self.input_dim = input_dim
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.hidden_size = rnn_size
        self.pack = PackSequence()
        self.unpack = PadPackedSequence()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=rnn_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, lengths):
        """LSTM forward
        Args:
            x (torch.Tensor):
                [B, S, F] Batch size x sequence length x feature size
                padded inputs
            lengths (torch.tensor):
                [B] Original lengths of each padded sequence in the batch
        Returns:
            torch.Tensor:
                [B, H] Batch size x hidden size lstm last timestep outputs
                2 x hidden_size if bidirectional
        """
        packed = self.pack(x, lengths)
        output, _ = self.lstm(packed)
        output, lengths = self.unpack(output)
        output = self.drop(output)

        rnn_all_outputs, last_timestep = self._final_output(output, lengths)
        # Use the last_timestep for classification / regression
        # Alternatively rnn_all_outputs can be used with an attention mechanism
        return last_timestep

    def _merge_bi(self, forward, backward):
        """Merge forward and backward states
        Args:
            forward (torch.Tensor): [B, L, H] Forward states
            backward (torch.Tensor): [B, L, H] Backward states
        Returns:
            torch.Tensor: [B, L, 2*H] Merged forward and backward states
        """
        return torch.cat((forward, backward), dim=-1)

    def _final_output(self, out, lengths):
        """Create RNN ouputs
        Collect last hidden state for forward and backward states
        Code adapted from https://stackoverflow.com/a/50950188
        Args:
            out (torch.Tensor): [B, L, num_directions * H] RNN outputs
            lengths (torch.Tensor): [B] Original sequence lengths
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (
                merged forward and backward states [B, L, H] or [B, L, 2*H],
                merged last forward and backward state [B, H] or [B, 2*H]
            )
        """

        if not self.bidirectional:
            return out, self._select_last_unpadded(out, lengths)

        forward, backward = (out[..., : self.hidden_size], out[..., self.hidden_size :])
        # Last backward corresponds to first token
        last_backward_out = backward[:, 0, :] if self.batch_first else backward[0, ...]
        # Last forward for real length or seq (unpadded tokens)
        last_forward_out = self._select_last_unpadded(forward, lengths)
        out = self._merge_bi(forward, backward)

        return out, self._merge_bi(last_forward_out, last_backward_out)

    def _select_last_unpadded(self, out, lengths):
        """Get the last timestep before padding starts
        Args:
            out (torch.Tensor): [B, L, H] Forward states
            lengths (torch.Tensor): [B] Original sequence lengths
        Returns:
            torch.Tensor: [B, H] Features for last sequence timestep
        """
        gather_dim = 1  # Batch first
        gather_idx = (
            (lengths - 1)  # -1 to convert to indices
            .unsqueeze(1)  # (B) -> (B, 1)
            .expand((-1, self.hidden_size))  # (B, 1) -> (B, H)
            # (B, 1, H) if batch_first else (1, B, H)
            .unsqueeze(gather_dim)
        )
        # Last forward for real length or seq (unpadded tokens)
        last_out = out.gather(gather_dim, gather_idx).squeeze(gather_dim)

        return last_out
    

class GenreClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, rnn_size=rnn_size, num_layers=num_layers, bidirectional=False, dropout=dropout):
        super(GenreClassifier, self).__init__()
        self.lstm = LSTMBackbone(input_dim, rnn_size, num_layers, bidirectional, dropout)
        #self.fc = nn.Linear(self.lstm.feature_size, num_classes)
        
        # Deeper classifier head with batch normalization and dropout
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm.feature_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, lengths):
        # LSTM backbone
        x = self.lstm(x, lengths)
        # Fully connected layer
        x = self.classifier(x)
        return x
    
    
        
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    num_batches = 0
    
    labels_all = torch.empty(0).to(device)
    preds_all = torch.empty(0).to(device)
    
    for (specs, labels, lengths) in train_loader:
        
        specs, labels = specs.to(device), labels.to(device)
        labels_all = torch.cat((labels_all, labels), dim=0)
    
        # Forward pass
        optimizer.zero_grad()
        output = model(specs, lengths)
        preds_all = torch.cat((preds_all, output), dim=0)
        
        loss = criterion(output, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute loss
        running_loss += loss.item()
        num_batches += 1
        
    train_loss = running_loss / num_batches
    return train_loss , preds_all, labels_all
        
        
def eval_epoch(model, val_loader, criterion, device=device):
    model.eval()
    running_loss = 0
    num_batches = 0
    
    y_pred = torch.empty(0).to(device)
    y_true = torch.empty(0).to(device)
    with torch.no_grad():
        for specs, labels, lengths in val_loader:
            specs, labels = specs.to(device), labels.to(device)
            
            logits = model(specs, lengths)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            # Predictions
            outputs = torch.argmax(logits, dim=1)
            y_pred = torch.cat((y_pred, outputs), dim=0)
            y_true = torch.cat((y_true, labels), dim=0)
            num_batches += 1
    valid_loss = running_loss / num_batches
    return valid_loss, y_pred, y_true


def train(train_loader, val_loader, num_epochs, device, overfit_batch=False, weight_decay=1e-4, patience=5):
    if not overfit_batch:
        input_dim = train_loader.dataset.feat_dim
        num_classes = len(np.unique(train_loader.dataset.labels))
        # Get unique class names
        unique_labels = np.unique(train_loader.dataset.labels)
        print(f"Unique labels: {unique_labels}")
        class_names = [train_loader.dataset.label_transformer.inverse(label) for label in range(num_classes)]
        print(f"Class names: {class_names}")
    else:
        input_dim = train_loader.dataset[0][0].shape[1]
        num_classes = 10
    
    model = GenreClassifier(
        input_dim=input_dim, num_classes=num_classes, rnn_size=rnn_size, 
        num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience, mode="min")
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
    # Initialize y_pred and y_true for confusion matrix
    y_pred = torch.empty(0, dtype=torch.int8)
    y_true = torch.empty(0, dtype=torch.int8)
    
    for epoch in range(num_epochs):
        training_loss, preds_all, labels_all = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, y_pred, y_true = eval_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}: train loss = {training_loss:.4f}, valid loss = {valid_loss:.4f}"
        )
        history['train_loss'].append(training_loss)
        history['val_loss'].append(valid_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Check if current model is the best
        if early_stopping.is_best(valid_loss) and not overfit_batch:
            print("New best model found! Saving checkpoint.")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
                'epoch': epoch,
            }, best_model_path)
            
        # If overfitting a batch, we care more about train loss going to zero
        if overfit_batch and training_loss < 1e-3:
            print("Successfully overfitted batch!")
            break
        
        if early_stopping.stop(valid_loss) and not overfit_batch:
            print("Early stopping...")
            break
    
    if not overfit_batch:
        # Return the best model and the training history
        checkpoint = torch.load(best_model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        valid_loss, y_pred, y_true = eval_epoch(model, val_loader, criterion)
        # Move tensors to CPU before converting to numpy
        y_pred = y_pred.cpu().numpy().ravel()
        y_true = y_true.cpu().numpy().ravel()
        #labels = np.arange(len(class_names))  # Create numeric labels
        cm = confusion_matrix(y_pred, y_true, normalize='true')
        plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Validation Confusion Matrix')
        # Print classification report with explicit handling
        try:
            # Convert integer labels to class names for better readability
            y_pred_classes = [class_names[int(i)] for i in y_pred]
            y_true_classes = [class_names[int(i)] for i in y_true]
            
            report = classification_report(
                y_true_classes, 
                y_pred_classes,
                digits=3,
                zero_division=0,
            )
            print("\nClassification Report:")
            print(report)
        except Exception as e:
            print(f"Error generating classification report: {str(e)}")
            # Fallback to basic metrics
            print("\nPer-class accuracies:")
            for i in range(num_classes):
                class_mask = (y_true == i)
                if np.any(class_mask):
                    class_acc = np.mean(y_pred[class_mask] == i)
                    print(f"{class_names[i]}: {class_acc:.3f}")
        print(f"Accuracy on validation set: {accuracy_score(y_true, y_pred):.4f}")
    else:
        # Calculate accuracy
        preds = torch.argmax(preds_all, dim=1)
        correct = (preds == labels_all).sum().item()
        total = len(labels_all)
        accuracy = correct / total
        print(f"Train accuracy: {accuracy:.4f}")
        # Convert integer labels to class names for better readability
        y_pred = y_pred.cpu().numpy().ravel()
        y_true = y_true.cpu().numpy().ravel()
        print(f"Accuracy on validation set: {accuracy_score(y_true, y_pred):.4f}")
    return model, history
             
def plot_training_history(history, title):
    """Plot training history including losses and learning rate.
    
    Args:
        history: Dictionary containing training history
    """

    # Plot losses
    plt.figure(figsize=(12, 8))
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/loss_plot_{}.png'.format(title))
    #plt.show()

def train_genre_classifier(dataset_path, feat_type='mel', batch_size=32, num_epochs=100, is_beat_sync=False):    
    
    if is_beat_sync:
        feat_type = 'beat'
    
    # Create datasets
    train_dataset = SpectrogramDataset(
        dataset_path, 
        class_mapping=CLASS_MAPPING,
        train=True,
        feat_type=feat_type
    )
    
    # Create data loaders
    train_loader, val_loader = torch_train_val_split(
        train_dataset,
        batch_train=batch_size,
        batch_eval=batch_size
    )
    
    # First verify the model can overfit a small batch
    print("Testing model with batch overfitting...")
    

    k = 3
    # Create a subset of the dataset of size k*batch_size and use this instead
    rng = np.random.default_rng(seed=seed)
    indices = rng.choice(np.arange(len(train_dataset)), size=k*batch_size, replace=False)
    train_ld_ov = Subset(train_dataset, indices)
    train_ov_loader = DataLoader(train_ld_ov, batch_size=batch_size,
                              pin_memory=True, shuffle=True)
    # Increase the number of epochs appropriately
    # total = epochs * len(dataset)
    #       = epochs * n_batches * batch_size
    #       = epochs * n_batches * k * (batch_size/k)
    # Thus, to keep roughly same total we do:
    num_epochs *= (batch_size // k) + 1
    # But we will use at most 200 epochs
    num_epochs = min(num_epochs, 200)
    print(f'Overfit Batch mode. The dataset now comprises of only {k} Batches. '
          f'Epochs increased to {num_epochs}.')
    _, history = train(
        train_ov_loader, 
        val_loader, 
        num_epochs=num_epochs,
        device=device,
        overfit_batch=True
    )
    
        
    # Plot training history
    plot_training_history(history, title=f"{feat_type}_overfit")
    
    # Reset model and train normally
    print("\nTraining full model...")
    model, history = train(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        device=device,
        overfit_batch=False
    )
    
    # Plot training history
    plot_training_history(history, title=f"{feat_type}_full")
    
    return model

def train_all_models(base_path, batch_size=32, num_epochs=50):
    """Train all required LSTM models for different input types"""
    
    # Dictionary to store results
    results = {}
    
    # γ) Train on mel spectrograms
    print("\n=== Training on Mel Spectrograms ===")
    mel_model = train_genre_classifier(
        base_path / "fma_genre_spectrograms",
        feat_type='mel',
        batch_size=batch_size,
        num_epochs=num_epochs,
    )
    results['mel'] = {'model': mel_model, 'type': 'regular'}
        
    # δ) Train on beat-synced spectrograms
    print("\n=== Training on Beat-Synced Spectrograms ===")
    beat_model = train_genre_classifier(
        base_path / "fma_genre_spectrograms_beat",
        feat_type='mel',
        batch_size=batch_size,
        num_epochs=num_epochs,
    )
    results['beat'] = {'model': beat_model, 'type': 'beat'}
        
    # ε) Train on chromagrams
    print("\n=== Training on Chromagrams ===")
    chroma_model = train_genre_classifier(
        base_path / "fma_genre_spectrograms",
        feat_type='chroma',
        batch_size=batch_size,
        num_epochs=num_epochs,
    )
    results['chroma'] = {'model': chroma_model, 'type': 'regular'}
        
    # ζ) Train on fused features (mel + chroma)
    print("\n=== Training on Fused Features ===")
    fused_model = train_genre_classifier(
        base_path / "fma_genre_spectrograms",
        feat_type='fused',
        batch_size=batch_size,
        num_epochs=num_epochs,
    )
    results['fused'] = {'model': fused_model, 'type': 'regular'}
        
    return results

def evaluate_models_on_test(base_path, model_dict, device, batch_size):
    """Evaluate trained models on both test sets"""
    results = {}
    criterion = nn.CrossEntropyLoss()
    
    # Create test datasets
    regular_test = SpectrogramDataset(
        base_path / "fma_genre_spectrograms",
        class_mapping=CLASS_MAPPING,
        train=False
    )
    beat_test = SpectrogramDataset(
        base_path / "fma_genre_spectrograms_beat",
        class_mapping=CLASS_MAPPING,
        train=False
    )
    
    # Create test loaders
    regular_loader, _ = torch_train_val_split(
        regular_test,
        batch_train=batch_size,
        batch_eval=batch_size,
        val_size=0.0
    )
    
    beat_loader, _ = torch_train_val_split(
        beat_test,
        batch_train=batch_size,
        batch_eval=batch_size,
        val_size=0.0
    )
    
    for model_name, model_info in model_dict.items():
        model = model_info['model']
        model.eval()
        
        # Use appropriate test loader based on model type
        test_loader = beat_loader if model_info['type'] == 'beat' else regular_loader
        description = f"{model_name.capitalize()} Model on {'Beat-Synced' if model_info['type'] == 'beat' else 'Regular'} Test Set"
        
        results[model_name] = evaluate_on_test_set(
            model, test_loader, criterion, device, description=description
        )
    
    return results


def evaluate_on_test_set(model, test_loader, criterion, device, description=""):
    """Evaluate a single model on a test set"""
    # Get unique class names
    num_classes = len(np.unique(test_loader.dataset.labels))
    class_names = [test_loader.dataset.label_transformer.inverse(label) for label in range(num_classes)]
    
    _ , y_pred , y_true = eval_epoch(model, test_loader, criterion, device)
    
    # Move tensors to CPU and ensure integers
    y_pred = y_pred.cpu().numpy().ravel()
    y_true = y_true.cpu().numpy().ravel()
    
    # Calculate all metrics
    report = classification_report(
        y_true, 
        y_pred,
        target_names=class_names,
        digits=3,
        zero_division=0,
        labels=np.arange(num_classes),  # Ensure all classes are present
        output_dict=True
    )
    
    # Print detailed results
    print(f"\nResults for {description}:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print("\nPer-class metrics:")
    print(classification_report(y_true, y_pred, target_names=class_names,zero_division=0))
    
    cm = confusion_matrix(y_pred, y_true, normalize='true')
    plot_confusion_matrix(cm, classes=test_loader.dataset.labels, normalize=True, title=f'{description} Validation Confusion Matrix')

    
    return {
        'accuracy': report['accuracy'],
        'per_class_metrics': {
            class_name: metrics for class_name, metrics in report.items()
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']
        },
        'macro_avg': report['macro avg'],
        'micro_avg': report['weighted avg'],  # sklearn uses 'weighted avg' for micro-averaging
        'confusion_matrix': cm
    }

if __name__ == '__main__':
    # Train all models
    results = train_all_models(base_path=base_path, num_epochs=num_epochs)
    
    # Evaluate on test sets
    test_results = evaluate_models_on_test(
        base_path=base_path,
        model_dict=results,
        device=device,
        batch_size=32
    )
    
        # Print summary comparison
    print("\nSummary Comparison:")
    for model_name, model_results in test_results.items():
        print(f"\n{'-'*20} {model_name} {'-'*20}")
        for test_set, metrics in model_results.items():
            print(f"\n{test_set} Test Set:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print("\nMacro-averaged metrics:")
            print(f"Precision: {metrics['macro_avg']['precision']:.4f}")
            print(f"Recall: {metrics['macro_avg']['recall']:.4f}")
            print(f"F1-score: {metrics['macro_avg']['f1-score']:.4f}")
            print("\nMicro-averaged metrics:")
            print(f"Precision: {metrics['micro_avg']['precision']:.4f}")
            print(f"Recall: {metrics['micro_avg']['recall']:.4f}")
            print(f"F1-score: {metrics['micro_avg']['f1-score']:.4f}")
