import torch.nn as nn

from convolution import CNNBackbone
from lstm import LSTMBackbone
from typing import Dict, Optional, Tuple, Union

import torch
import os

def load_backbone_from_checkpoint(model, checkpoint_path):
    """
    A simplified function to load weights from a checkpoint into a backbone model.
    Handles both complete model checkpoints and backbone-only checkpoints.
    
    Args:
        model (nn.Module): The backbone model to load weights into
        checkpoint_path (str): Path to the checkpoint file
        
    Returns:
        nn.Module: The model with loaded weights
    """
    # Ensure checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # If checkpoint contains full model state
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Extract backbone weights if they exist
        backbone_state_dict = {}
        for key, value in state_dict.items():
            # Remove 'backbone.' prefix if it exists
            if key.startswith('backbone.'):
                backbone_state_dict[key.replace('backbone.', '')] = value
            else:
                backbone_state_dict[key] = value
    else:
        backbone_state_dict = checkpoint
    
    # Load the weights
    model.load_state_dict(backbone_state_dict, strict=False)
    model.eval()
    
    return model
class Classifier(nn.Module):
    def __init__(self, num_classes, backbone, load_from_checkpoint=None):
        """
        backbone (nn.Module): The nn.Module to use for spectrogram parsing
        num_classes (int): The number of classes
        load_from_checkpoint (Optional[str]): Use a pretrained checkpoint to initialize the model
        """
        super(Classifier, self).__init__()
        self.backbone = backbone  # An LSTMBackbone or CNNBackbone
        if load_from_checkpoint is not None:
            self.backbone = load_backbone_from_checkpoint(
                self.backbone, load_from_checkpoint
            )
        self.num_classes = num_classes
        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()  # Loss function for classification

    def forward(self, x: torch.Tensor, 
                targets: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass supporting both training and inference modes.
        
        Args:
            x: Input tensor
            targets: Optional target labels (for training)
            lengths: Optional sequence lengths (for LSTM)
            
        Returns:
            During training: (loss, logits)
            During inference: logits only
        """
        # Get features from backbone, handling LSTM vs non-LSTM case
        if self.is_lstm and lengths is not None:
            feats = self.backbone(x, lengths)
        else:
            feats = self.backbone(x)
            
        logits = self.output_layer(feats)
        
        if targets is not None:
            loss = self.criterion(logits, targets)
            return loss, logits
        return logits
    
    def get_backbone_name(self):
        return self.backbone.get_backbone_name()
    

class Regressor(nn.Module):
    def __init__(self, backbone, load_from_checkpoint=None):
        """
        backbone (nn.Module): The nn.Module to use for spectrogram parsing
        load_from_checkpoint (Optional[str]): Use a pretrained checkpoint to initialize the model
        """
        super(Regressor, self).__init__()
        self.backbone = backbone  # An LSTMBackbone or CNNBackbone
        if load_from_checkpoint is not None:
            self.backbone = load_backbone_from_checkpoint(
                self.backbone, load_from_checkpoint
            )
        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, 1)
        self.criterion = nn.MSELoss()  # Loss function for regression

    def forward(self, x: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass supporting both training and inference modes.
        """
        # Get features from backbone, handling LSTM vs non-LSTM case
        if self.is_lstm and lengths is not None:
            feats = self.backbone(x, lengths)
        else:
            feats = self.backbone(x)
            
        predictions = self.output_layer(feats)
        predictions = predictions.squeeze(-1)  # Remove the last dimension to match target shape
        
        if targets is not None:
            loss = self.criterion(predictions.float(), targets.float())
            return loss, predictions
        return predictions
    
    def get_backbone_name(self):
        return self.backbone.get_backbone_name()


class MultitaskRegressor(nn.Module):
    def __init__(self, backbone, load_from_checkpoint=None):
        """
        A regressor that handles multiple regression tasks simultaneously while maintaining
        compatibility with the existing training infrastructure.
        
        The model predicts valence, arousal, and danceability using a shared backbone
        and separate prediction heads for each task.
        """
        super(MultitaskRegressor, self).__init__()
        self.backbone = backbone
        if load_from_checkpoint is not None:
            self.backbone = load_backbone_from_checkpoint(
                self.backbone, load_from_checkpoint
            )
        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        
        # Create separate output layers for each task
        self.valence_layer = nn.Linear(self.backbone.feature_size, 1)
        self.arousal_layer = nn.Linear(self.backbone.feature_size, 1)
        self.danceability_layer = nn.Linear(self.backbone.feature_size, 1)
        
        self.criterion = nn.MSELoss()
        
        # Weights for each task to balance the loss
        self.task_weights = {
            'valence': 1.5,
            'arousal': 0.8,
            'danceability': 0.9              
        }
    
    def forward(self, x: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass compatible with train_with_eval function.
        
        Args:
            x: Input spectrograms of shape (batch_size, time_steps, features)
            targets: Target values of shape (batch_size, 3) where each column represents
                    valence, arousal, and danceability respectively
            lengths: Optional sequence lengths for LSTM processing
            
        Returns:
            During training (targets provided):
                - tuple (loss, predictions)
            During inference (no targets):
                - predictions only
        """
        # Get features from backbone
        if self.is_lstm and lengths is not None:
            features = self.backbone(x, lengths)
        else:
            features = self.backbone(x)
        
        # Get predictions for each task
        valence_pred = self.valence_layer(features).squeeze(-1)
        arousal_pred = self.arousal_layer(features).squeeze(-1)
        danceability_pred = self.danceability_layer(features).squeeze(-1)
        
        # Stack predictions to match target format
        predictions = torch.stack([valence_pred, arousal_pred, danceability_pred], dim=1)
        
        if targets is not None:
            # Calculate individual losses
            valence_loss = self.criterion(valence_pred, targets[:, 0])
            arousal_loss = self.criterion(arousal_pred, targets[:, 1])
            danceability_loss = self.criterion(danceability_pred, targets[:, 2])
            
            # Apply task weights
            total_loss = (
                self.task_weights['valence'] * valence_loss +
                self.task_weights['arousal'] * arousal_loss +
                self.task_weights['danceability'] * danceability_loss
            )
            
            return total_loss, predictions
        
        return predictions
    
    def get_backbone_name(self):
        return f"Multitask-{self.backbone.get_backbone_name()}"