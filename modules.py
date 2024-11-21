import torch.nn as nn

from convolution import CNNBackbone
from lstm import LSTMBackbone


def load_backbone_from_checkpoint(model, checkpoint_path):
    raise NotImplementedError("You need to implement this")

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
    # TODO: Implement this like the Classifier
    pass
