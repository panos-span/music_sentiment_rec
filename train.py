import torch
import os

from dataset import SpectrogramDataset, torch_train_val_split, CLASS_MAPPING
from convolution import CNNBackbone
from lstm import LSTMBackbone
from modules import Classifier
from train_utils import train

class Training(object):
    def __init__(self, model, train_loader, val_loader, optimizer, epochs, save_path, device, overfit_batch):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.epochs = epochs
        self.save = save_path
        self.device = device
        self.overfit_batch = overfit_batch

    def train_with_eval(self):
        train(self.model, self.train_loader, self.val_loader, self.optimizer, self.epochs, self.save, self.device, self.overfit_batch)
        ## to implement to return losses
        return None


if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Data Params
    dataset_path = './data/fma_genre_spectrograms'
    batch_size = 32
    max_length = 150
    feat_type = 'mel'
    train_data = True

    # Create datasets
    train_dataset = SpectrogramDataset(
        dataset_path, 
        class_mapping=CLASS_MAPPING,
        train=train_data,
        max_length=max_length,
        feat_type=feat_type
    )
    
    # Create data loaders
    train_loader, val_loader = torch_train_val_split(
        train_dataset,
        batch_train=batch_size,
        batch_eval=batch_size,
    )

    # Training Hyperparams
    epochs = 100
    overfit_batch = True

    # Directory to save checkpoints
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    CP_PATH = 'checkpoint.pth'
    cp_path = os.path.join(checkpoint_dir, CP_PATH)

    # LSTM Hyperparams
    input_dim = train_dataset.feat_dim
    dropout = 0.0
    rnn_size = 4
    num_layers = 1
    bidirectional = False
    num_classes = 10

    # init LSTM
    backbone = LSTMBackbone(input_dim, rnn_size, num_layers, bidirectional, dropout)
    model = Classifier(num_classes, backbone)

    # Optimizer
    lr = 1e-4
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # init training
    lstm_ = Training(model, train_loader, val_loader, optimizer, epochs, cp_path, DEVICE, overfit_batch=overfit_batch)
    # start training
    lstm_.train_with_eval()


    # CNN Hyperparams
    input_shape = (train_dataset.max_length, train_dataset.feat_dim)
    cnn_in_channels = 1
    cnn_filters = [32, 64, 128, 256]
    cnn_out_feature_size = 1000

    # init CNN
    backbone = CNNBackbone(input_shape, cnn_in_channels, cnn_filters, cnn_out_feature_size)
    model = Classifier(num_classes, backbone)
    
    # init training
    cnn_ = Training(model, train_loader, val_loader, optimizer, epochs, cp_path, DEVICE, overfit_batch=overfit_batch)
    # start training
    cnn_.train_with_eval()