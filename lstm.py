# A full LSTM implementation is provided
# You can use this or the one you implemented in the second lab

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataset import SpectrogramDataset, torch_train_val_split, CLASS_MAPPING
import copy
import matplotlib.pyplot as plt
import numpy as np
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

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
        rnn_size=128,
        num_layers=1,
        bidirectional=False,
        dropout=0.1,
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
            out (torch.Tensor): [B, L, H] Fprward states
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
    def __init__(self, input_dim, num_classes, rnn_size=128, num_layers=2, dropout=0.2):
        super(GenreClassifier, self).__init__()
        
        self.lstm = LSTMBackbone(input_dim, rnn_size=rnn_size, num_layers=num_layers, dropout=dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm.feature_size, rnn_size),     
            nn.ReLU(),
            nn.Linear(rnn_size, num_classes)
        )
        
    
    def forward(self, x, lengths):
        # Get lstm output
        x = self.lstm(x, lengths)
        # Pass lstm output to classifier
        x = self.classifier(x)
        return x
    

def train_epoch(model, train_loader, criterion, optimizer, device, overfit_batch=False):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # If overfitting a batch, we'll only use the first few batches
    batch_limit = 3 if overfit_batch else float('inf')
    
    for batch_idx, (specs, labels, lengths) in enumerate(train_loader):
        if batch_idx >= batch_limit and overfit_batch:
            break
            
        specs = specs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(specs, lengths)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()
        
    avg_loss = total_loss / (batch_idx + 1)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for specs, labels, lengths in val_loader:
            specs = specs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            outputs = model(specs, lengths)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def train(model, train_loader, val_loader, num_epochs, device, overfit_batch=False, weight_decay=1e-4, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, mode="min")
    best_model = None
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, overfit_batch
        )
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping check
        if early_stopping.is_best(val_loss):
            best_model = copy.deepcopy(model.state_dict())
            
        # If overfitting a batch, we care more about train loss going to zero
        if overfit_batch and train_loss < 0.01:
            print("Successfully overfitted batch!")
            break
            
        if early_stopping.stop(val_loss) and not overfit_batch:
            print("Early stopping triggered")
            break
    
    # Restore best model
    if best_model is not None and not overfit_batch:
        model.load_state_dict(best_model)
    
    return model, history


def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def train_genre_classifier(dataset_path, feat_type='mel', batch_size=32, num_epochs=50):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    # Create model
    input_dim = train_dataset.feat_dim
    num_classes = len(np.unique(train_dataset.labels))
    model = GenreClassifier(input_dim, num_classes).to(device)
    
    # First verify the model can overfit a small batch
    print("Testing model with batch overfitting...")
    _, _ = train(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=50,
        device=device,
        overfit_batch=True
    )
    
    # Reset model and train normally
    print("\nTraining full model...")
    model = GenreClassifier(input_dim, num_classes).to(device)
    model, history = train(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        device=device,
        overfit_batch=False
    )
    
    # Plot training history
    plot_training_history(history)
    
    return model

def train_model(dataset_path, feat_type, batch_size, num_epochs, device):
    """Train a model on specified feature type"""
    # Create dataset
    dataset = SpectrogramDataset(
        dataset_path,
        class_mapping=CLASS_MAPPING,
        train=True,
        feat_type=feat_type
    )
    
    # Create data loaders
    train_loader, val_loader = torch_train_val_split(
        dataset,
        batch_train=batch_size,
        batch_eval=batch_size
    )
    
    # Create model
    input_dim = dataset.feat_dim
    num_classes = len(np.unique(dataset.labels))
    
    model = GenreClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        rnn_size=128,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    # Train model
    model, history = train(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        device=device
    )
    
    return {'model': model, 'history': history}


def train_all_models(base_path, batch_size=32, num_epochs=50):
    """Train all required LSTM models for different input types"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dictionary to store results
    results = {}
    
    # γ) Train on mel spectrograms
    print("\n=== Training on Mel Spectrograms ===")
    mel_model = train_model(
        base_path + "/fma_genre_spectrograms",
        feat_type='mel',
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device
    )
    results['mel'] = mel_model
    
    # δ) Train on beat-synced spectrograms
    print("\n=== Training on Beat-Synced Spectrograms ===")
    beat_model = train_model(
        base_path + "/fma_genre_spectrograms_beat",
        feat_type='mel',
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device
    )
    results['beat'] = beat_model
    
    # ε) Train on chromagrams
    print("\n=== Training on Chromagrams ===")
    chroma_model = train_model(
        base_path + "/fma_genre_spectrograms",
        feat_type='chroma',
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device
    )
    results['chroma'] = chroma_model
    
    # ζ) Train on fused features (mel + chroma)
    print("\n=== Training on Fused Features ===")
    fused_model = train_model(
        base_path + "/fma_genre_spectrograms",
        feat_type='fused',
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device
    )
    results['fused'] = fused_model
    
    return results