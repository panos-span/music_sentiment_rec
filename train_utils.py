import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from plot_confusion_matrix import plot_confusion_matrix
from modules import Classifier

class EarlyStopper:
    def __init__(self, model, save_path, patience=1, min_delta=0):
        self.model = model
        self.save_path = save_path
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(self.model.state_dict(), self.save_path)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y, lengths in train_loader:        
        loss, logits = model(x.float().to(device), y.to(device), lengths.to(device))
        # prepare
        optimizer.zero_grad()
        # backward
        loss.backward()
        # optimizer step
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)    
    return avg_loss


def validate_one_epoch(model, val_loader, device):    
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y, lengths in val_loader:
            loss, logits = model(x.float().to(device), y.to(device), lengths.to(device))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)    
    return avg_loss



def overfit_with_a_couple_of_batches(model, train_loader, optimizer, device, epochs: int):
    """
    Train the model by overfitting on 3 batches to verify the network can learn.
    This helps ensure gradients flow properly through the network.
    
    Args:
        model: The neural network model
        train_loader: DataLoader containing training data
        optimizer: The optimizer for training
        device: Device to run training on (cuda/cpu)
        epochs: Number of epochs to train
    """
    print('Training in overfitting mode (3 batches)...')
    
    # Get the first 3 batches
    train_iter = iter(train_loader)
    batches = []
    for _ in range(3):
        try:
            batch = next(train_iter)
            batches.append((
                batch[0].float().to(device),  # x
                batch[1].to(device),          # y
                batch[2].to(device)           # lengths
            ))
        except StopIteration:
            print("Warning: Could not get 3 full batches, using available batches only")
            break
    
    if not batches:
        raise RuntimeError("No batches available for training")
    
    model.train()
    
    total_avg_per_epoch = 0
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(epochs):
        total_loss = 0
        
        # Train on each of the saved batches
        for batch_idx, (x, y, lengths) in enumerate(batches):
            # Forward pass
            loss, logits = model(x, y, lengths) 
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss across batches
        avg_loss = total_loss / len(batches)
        history['train_loss'].append(avg_loss)
        
        total_avg_per_epoch += avg_loss
        
        # Print progress
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch + 1}, Average loss across {len(batches)} batches: {avg_loss:.6f}')

        # Validate the model
        val_loss = validate_one_epoch(model, train_loader, device)
        history['val_loss'].append(val_loss)
    
    
    # Calculate average loss across all epochs
    total_avg_per_epoch /= epochs
    print(f'Average loss across all epochs: {total_avg_per_epoch:.6f}')
    plot_training_history(history, title="Overfitting Training and Validation Loss")
    return model
    

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


def train(model, train_loader, val_loader, optimizer, epochs, save_path, device, overfit_batch):
    # Get unique class names
    #num_classes = len(np.unique(train_loader.dataset.labels))
    #unique_labels = np.unique(train_loader.dataset.labels)
    #print(f"Unique labels: {unique_labels}")
    #class_names = [train_loader.dataset.label_transformer.inverse(label) for label in range(num_classes)]
    history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}

    if overfit_batch:
       train_loss = overfit_with_a_couple_of_batches(model, train_loader, optimizer, device, epochs)
    else:
        print(f'Training started for model {save_path.replace(".pth", "")}...')
        early_stopper = EarlyStopper(model, save_path, patience=5)
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            validation_loss = validate_one_epoch(model, val_loader, device)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(validation_loss)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            if epoch== 0 or (epoch+1) % 5==0:
                print(f'Epoch {epoch+1}/{epochs}, Loss at training set: {train_loss}\n\tLoss at validation set: {validation_loss}')          
            
            if early_stopper.early_stop(validation_loss):
                print('Early Stopping was activated.')
                print(f'Epoch {epoch+1}/{epochs}, Loss at training set: {train_loss}\n\tLoss at validation set: {validation_loss}')
                print('Training has been completed.\n')
                break
            
    if not overfit_batch:
        # Return the best model and history
        best_model = Classifier(model.num_classes, model.backbone)
        best_model.load_state_dict(torch.load(save_path),weights_only=True)
        plot_training_history(history, title="Training and Validation Loss")    
    else:
        best_model = model    
    return best_model
    
    if not overfit_batch:
        checkpoint = torch.load(save_path, weights_only=True)
        model.load_state_dict(checkpoint)
        valid_loss, y_true, y_pred = validate_one_epoch(model, val_loader, device)
        # Move to cpu
        y_true = y_true.cpu().numpy().ravel()
        y_pred = y_pred.cpu().numpy().ravel()
        
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix', normalize=True)
        
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
            for i in range(model.num_classes):
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
        
    