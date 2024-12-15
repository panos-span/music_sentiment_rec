import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from plot_confusion_matrix import plot_confusion_matrix
from modules import Classifier
import os

class EarlyStopper:
    """
    Early stopping handler that saves model checkpoints and stops training when 
    validation loss stops improving.
    """
    def __init__(self, model, save_path, patience=1, min_delta=0):
        """
        Initialize the early stopper.
        
        Args:
            model: The model being trained
            save_path: Where to save the model checkpoint
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in loss to qualify as an improvement
        """
        self.model = model
        self.save_path = save_path
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def save_checkpoint(self, model_state, validation_loss, epoch, optimizer_state=None):
        """
        Save a checkpoint of the model.
        
        Args:
            model_state: Current state of the model
            validation_loss: Current validation loss
            epoch: Current epoch number
            optimizer_state: Optional optimizer state
        """
        checkpoint = {
            'model_state_dict': model_state,
            'epoch': epoch,
            'validation_loss': validation_loss
        }
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        torch.save(checkpoint, self.save_path)
    
    def early_stop(self, validation_loss, epoch=None, optimizer_state=None):
        """
        Check if training should stop and save checkpoint if loss improved.
        
        Args:
            validation_loss: Current validation loss
            epoch: Current epoch number (optional)
            optimizer_state: Current optimizer state (optional)
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if validation_loss < self.min_validation_loss:
            # Save checkpoint since we found a better model
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.save_checkpoint(
                self.model.state_dict(),
                validation_loss,
                epoch,
                optimizer_state
            )
            return False
        
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



def overfit_with_a_couple_of_batches(model, train_loader, optimizer, device, epochs: int, title=None):
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
    
    total_avg_per_epoch = 0
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(epochs):
        model.train()  # Move inside epoch loop
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
    plot_training_history(history, title=title)
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
    plt.title(f'{title}')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/{}.png'.format(title))


def train(model, train_loader, val_loader, optimizer, epochs, save_path, device, overfit_batch, title):
    """
    Modified training function to work with the improved checkpoint system.
    """
    model.train()
    history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
    
    if overfit_batch:
        train_loss = overfit_with_a_couple_of_batches(model, train_loader, optimizer, device, epochs=epochs, title=title)
        return model
    
    print(f'Training started for model {os.path.basename(save_path).replace(".pth", "")}...')
    early_stopper = EarlyStopper(model, save_path, patience=5)
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        validation_loss = validate_one_epoch(model, val_loader, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(validation_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train loss: {train_loss:.4f}, Val loss: {validation_loss:.4f}')
        
        if early_stopper.early_stop(validation_loss, epoch, optimizer.state_dict()):
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    # Load the best model
    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded best model from epoch {checkpoint["epoch"] + 1} with validation loss: {checkpoint["validation_loss"]:.4f}')
    
    if not overfit_batch:
        plot_training_history(history, title=title)
    
    return model
