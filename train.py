import torch
import os

from dataset import SpectrogramDataset, torch_train_val_split, CLASS_MAPPING
from convolution import CNNBackbone
from lstm import LSTMBackbone
from modules import Classifier, Regressor
from train_utils import train
from evaluation import evaluate, predict
from ast_nn import ASTBackbone

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
        model =  train(self.model, self.train_loader, self.val_loader, self.optimizer, self.epochs, self.save, self.device, self.overfit_batch)
        results = evaluate(model, self.val_loader, self.device)
        return results


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
    
    # init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # init training
    cnn_ = Training(model, train_loader, val_loader, optimizer, epochs, cp_path, DEVICE, overfit_batch=overfit_batch)
    # start training
    cnn_.train_with_eval()

    # AST Hyperparams
    input_fdim = 128
    input_tdim = 150
    feature_size = 100

    # init AST
    backbone = ASTBackbone(input_fdim=input_fdim, input_tdim=input_tdim, feature_size=feature_size)
    model = Classifier(10, backbone)
    
    # init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # init training
    ast_ = Training(model, train_loader, val_loader, optimizer, epochs, cp_path, DEVICE, overfit_batch=overfit_batch)
    # start training
    ast_.train_with_eval()
    
    # Regressor
    
    tasks = ['valence', 'energy', 'danceability']
    
    test_results = {
        'lstm': [],
        'cnn': [],
        'ast': []
    }
    
    for task in tasks:
        label_index = tasks.index(task) + 1

        train_dataset = SpectrogramDataset(
            dataset_path, class_mapping=CLASS_MAPPING,
            train=True, feat_type=feat_type, max_length=max_length, regression=label_index, test=True
        )

        train_loader, val_loader, test_loader = torch_train_val_split(
            train_dataset, batch_train=batch_size, batch_eval=batch_size
        )
        
        # Create LSTM Backbone Regressor
        # LSTM Hyperparams
        input_dim = train_dataset.feat_dim
        dropout = 0.0
        rnn_size = 4
        num_layers = 1
        bidirectional = False
        
        # init LSTM
        backbone = LSTMBackbone(input_dim, rnn_size, num_layers, bidirectional, dropout)
        model = Regressor(backbone)
        
        # init optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # init training
        lstm_regressor = Training(model, train_loader, val_loader, optimizer, epochs, cp_path, DEVICE, overfit_batch=overfit_batch)
        # start training
        lstm_regressor.train_with_eval()
        
        # Create CNN Backbone Regressor
        # CNN Hyperparams
        input_shape = (train_dataset.max_length, train_dataset.feat_dim)
        cnn_in_channels = 1
        cnn_filters = [32, 64, 128, 256]
        cnn_out_feature_size = 1000
        
        # init CNN
        backbone = CNNBackbone(input_shape, cnn_in_channels, cnn_filters, cnn_out_feature_size)
        model = Regressor(backbone)
        
        # init optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # init training
        cnn_regressor = Training(model, train_loader, val_loader, optimizer, epochs, cp_path, DEVICE, overfit_batch=overfit_batch)
        cnn_regressor.train_with_eval()
        
        # Create AST Backbone Regressor
        # AST Hyperparams
        input_fdim = 128
        input_tdim = 150
        feature_size = 100
        
        # init AST
        backbone = ASTBackbone(input_fdim=input_fdim, input_tdim=input_tdim, feature_size=feature_size)
        model = Regressor(backbone)
        
        # init optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # init training
        ast_regressor = Training(model, train_loader, val_loader, optimizer, epochs, cp_path, DEVICE, overfit_batch=overfit_batch)
        ast_regressor.train_with_eval()
        
        
        