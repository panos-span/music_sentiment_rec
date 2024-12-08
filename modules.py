import torch.nn as nn

from convolution import CNNBackbone
from lstm import LSTMBackbone


import torch
import os

def load_backbone_from_checkpoint(model, checkpoint_path):
    """
    Load weights from a checkpoint file into the backbone model.
    
    Args:
        model (nn.Module): The backbone model to load weights into
        checkpoint_path (str): Path to the checkpoint file
        
    Returns:
        nn.Module: The model with loaded weights
        
    Raises:
        FileNotFoundError: If checkpoint_path doesn't exist
        RuntimeError: If checkpoint is incompatible with model architecture
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict, handling different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'backbone' in checkpoint:
            state_dict = checkpoint['backbone']
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Checkpoint format not recognized")
    
    # Filter only backbone parameters
    backbone_state_dict = {}
    for k, v in state_dict.items():
        # Handle different key formats
        if k.startswith('backbone.'):
            # Remove 'backbone.' prefix if it exists
            backbone_state_dict[k.replace('backbone.', '')] = v
        elif k.startswith('module.backbone.'):
            # Handle DataParallel format
            backbone_state_dict[k.replace('module.backbone.', '')] = v
        elif k.startswith('module.'):
            # Handle DataParallel format without backbone prefix
            backbone_state_dict[k.replace('module.', '')] = v
        else:
            backbone_state_dict[k] = v
            
    # Try to load the filtered state dict
    try:
        model.load_state_dict(backbone_state_dict, strict=True)
    except RuntimeError as e:
        # If strict loading fails, try non-strict loading
        missing_keys, unexpected_keys = model.load_state_dict(backbone_state_dict, strict=False)
        print(f"Warning: Non-strict loading performed.")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
            
    # Set model to evaluation mode
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

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        logits = self.output_layer(feats)
        loss = self.criterion(logits, targets)
        return loss, logits
    

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

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        out = self.output_layer(feats)
        loss = self.criterion(out.float(), targets.float())
        return loss, out


class MultitaskRegressor(nn.Module):
    def __init__(self, backbone, load_from_checkpoint=None):
        """
        An extended version of the Regressor that handles multiple regression tasks simultaneously
        
        Args:
            backbone (nn.Module): The nn.Module to use for spectrogram parsing
            load_from_checkpoint (Optional[str]): Use a pretrained checkpoint to initialize the model
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
        
        # Task weights to balance the losses
        # These can be adjusted based on the scale of each task's values
        self.task_weights = {
            'valence': 1.0,
            'arousal': 1.0,
            'danceability': 1.0
        }
    
    def compute_multitask_loss(self, valence_out, arousal_out, danceability_out, 
                             valence_targets, arousal_targets, danceability_targets):
        """
        Compute the weighted sum of losses for all three tasks
        
        Args:
            *_out: Model predictions for each task
            *_targets: Ground truth values for each task
            
        Returns:
            total_loss: Weighted sum of individual task losses
            individual_losses: Dictionary containing individual losses for monitoring
        """
        valence_loss = self.criterion(valence_out.float(), valence_targets.float())
        arousal_loss = self.criterion(arousal_out.float(), arousal_targets.float())
        danceability_loss = self.criterion(danceability_out.float(), danceability_targets.float())
        
        # Apply task weights
        weighted_valence_loss = self.task_weights['valence'] * valence_loss
        weighted_arousal_loss = self.task_weights['arousal'] * arousal_loss
        weighted_danceability_loss = self.task_weights['danceability'] * danceability_loss
        
        # Compute total loss
        total_loss = weighted_valence_loss + weighted_arousal_loss + weighted_danceability_loss
        
        # Return both total loss and individual losses for monitoring
        individual_losses = {
            'valence': valence_loss.item(),
            'arousal': arousal_loss.item(),
            'danceability': danceability_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, individual_losses
    
    def forward(self, x, targets, lengths):
        """
        Forward pass handling multiple targets
        
        Args:
            x: Input features
            targets: Dictionary containing targets for each task
            lengths: Sequence lengths for LSTM
            
        Returns:
            total_loss: Combined loss from all tasks
            outputs: Dictionary containing predictions for each task
            individual_losses: Dictionary containing individual task losses
        """
        # Get backbone features
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        
        # Get predictions for each task
        valence_out = self.valence_layer(feats)
        arousal_out = self.arousal_layer(feats)
        danceability_out = self.danceability_layer(feats)
        
        # Compute combined loss
        total_loss, individual_losses = self.compute_multitask_loss(
            valence_out, arousal_out, danceability_out,
            targets['valence'], targets['arousal'], targets['danceability']
        )
        
        # Package outputs
        outputs = {
            'valence': valence_out,
            'arousal': arousal_out,
            'danceability': danceability_out
        }
        
        return total_loss, outputs, individual_losses