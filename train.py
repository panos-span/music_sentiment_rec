import torch
import os

from dataset import SpectrogramDataset, torch_train_val_split, CLASS_MAPPING
from convolution import CNNBackbone
from lstm import LSTMBackbone
from modules import Classifier, Regressor, MultitaskRegressor
from train_utils import train
from evaluation import evaluate, predict
from pathlib import Path
from ast_nn import ASTBackbone


class Training(object):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs,
        save_path,
        device,
        overfit_batch,
        feat_type=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.epochs = epochs
        self.save = save_path
        self.device = device
        self.overfit_batch = overfit_batch
        self.loss_plot_title = f"{self.model.get_backbone_name()} with {overfit_batch} overfitting with {feat_type} feat"
        self.feat_type = feat_type

    def train_with_eval(self):
        print(self.loss_plot_title)
        model = train(
            self.model,
            self.train_loader,
            self.val_loader,
            self.optimizer,
            self.epochs,
            self.save,
            self.device,
            self.overfit_batch,
            self.loss_plot_title,
        )
        results = evaluate(model, self.val_loader, self.device, title=f"Confusion Matrix {self.loss_plot_title}")
        return results
    
def setup_regression_training(task, dataset_path, feat_type, max_length, batch_size, 
                            device, lr, weight_decay, epochs, cp_path, overfit_batch):
    """
    Set up and train regression models for a specific task.
    """
    tasks = ['valence', 'energy', 'danceability']
    label_index = tasks.index(task) + 1
    
    # Create dataset with proper regression index
    train_dataset = SpectrogramDataset(
        dataset_path,
        class_mapping=None,  # Not needed for regression
        train=True,
        feat_type=feat_type,
        max_length=max_length,
        regression=label_index
    )
    
    # Create data loaders with consistent batch sizes
    train_loader, val_loader, test_loader = torch_train_val_split(
        train_dataset,
        batch_train=batch_size,
        batch_eval=batch_size,
        test=True,
        val_size=0.2
    )
    
    results = {}
    
    # Train LSTM Regressor
    print(f"\nTraining LSTM Regressor for {task}")
    lstm_backbone = LSTMBackbone(
        input_dim=train_dataset.feat_dim,
        rnn_size=4,
        num_layers=1,
        bidirectional=False,
        dropout=0.0
    )
    lstm_model = Regressor(lstm_backbone).to(device)
    lstm_optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=lr, weight_decay=weight_decay)
    lstm_trainer = Training(
        model=lstm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=lstm_optimizer,
        epochs=epochs,
        save_path=cp_path,
        device=device,
        overfit_batch=overfit_batch,
        feat_type=feat_type
    )
    results['lstm'] = lstm_trainer.train_with_eval()
    
    # Train CNN Regressor
    print(f"\nTraining CNN Regressor for {task}")
    cnn_backbone = CNNBackbone(
        input_dims=(train_dataset.max_length, train_dataset.feat_dim),
        in_channels=1,
        filters=[32, 64, 128, 256],
        feature_size=1000
    )
    cnn_model = Regressor(cnn_backbone).to(device)
    cnn_optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=lr, weight_decay=weight_decay)
    cnn_trainer = Training(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=cnn_optimizer,
        epochs=epochs,
        save_path=cp_path,
        device=device,
        overfit_batch=overfit_batch,
        feat_type=feat_type
    )
    results['cnn'] = cnn_trainer.train_with_eval()
    
    # Train AST Regressor
    print(f"\nTraining AST Regressor for {task}")
    ast_backbone = ASTBackbone(
        input_fdim=128,
        input_tdim=150,
        feature_size=100
    )
    ast_model = Regressor(ast_backbone).to(device)
    ast_optimizer = torch.optim.AdamW(ast_model.parameters(), lr=lr, weight_decay=weight_decay)
    ast_trainer = Training(
        model=ast_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=ast_optimizer,
        epochs=epochs,
        save_path=cp_path,
        device=device,
        overfit_batch=overfit_batch,
        feat_type=feat_type
    )
    results['ast'] = ast_trainer.train_with_eval()
    
    return results

# Update the main training loop
def train_regression_models(dataset_path, feat_type, max_length, batch_size, 
                          device, lr, weight_decay, epochs, cp_path, overfit_batch):
    tasks = ['valence', 'energy', 'danceability']
    all_results = {}
    
    for task in tasks:
        print(f"\nTraining regression models for {task}")
        all_results[task] = setup_regression_training(
            task=task,
            dataset_path=dataset_path,  # Use the correct dataset path
            feat_type=feat_type,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            cp_path=cp_path,
            overfit_batch=overfit_batch
        )
    
    return all_results

def setup_transfer_learning(source_model_path, dataset_path, feat_type='mel', task='valence',
                          lr=1e-5, batch_size=32, epochs=20, device='cuda'):
    """
    Set up and perform transfer learning with proper weight mapping.
    """
    # Load the source model checkpoint
    print(f"Loading pre-trained model from {source_model_path}")
    checkpoint = torch.load(source_model_path, map_location=device, weights_only=True)
    
    # Extract model state dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            source_state = checkpoint['model_state_dict']
        else:
            source_state = checkpoint
    
    # Create new state dict with corrected keys
    corrected_state = {}
    for key, value in source_state.items():
        # Remove 'backbone.' prefix if it exists
        if key.startswith('backbone.'):
            new_key = key.replace('backbone.', '')
            corrected_state[new_key] = value
        else:
            corrected_state[key] = value
    
    # Create dataset for sentiment regression
    tasks = ['valence', 'energy', 'danceability']
    label_index = tasks.index(task) + 1
    
    train_dataset = SpectrogramDataset(
        dataset_path,
        class_mapping=None,
        train=True,
        feat_type=feat_type,
        max_length=150,
        regression=label_index
    )
    
    # Create data loaders
    train_loader, val_loader, _ = torch_train_val_split(
        train_dataset,
        batch_train=batch_size,
        batch_eval=batch_size
    )
    
    # Initialize the AST model for regression
    backbone = ASTBackbone(
        input_fdim=128,
        input_tdim=150,
        feature_size=100
    )
    model = Regressor(backbone)
    
    # Remove the classification head weights
    keys_to_remove = [k for k in corrected_state.keys() 
                     if k.startswith('output_layer') or 
                        k.startswith('fc') or 
                        k.startswith('head')]
    for key in keys_to_remove:
        del corrected_state[key]
    
    # Load the pre-trained weights into the backbone
    print("Transferring weights from pre-trained model...")
    missing_keys, unexpected_keys = model.backbone.load_state_dict(
        corrected_state,
        strict=False
    )
    
    if missing_keys:
        print("\nMissing keys in target model:")
        for key in missing_keys:
            print(f"- {key}")
    
    if unexpected_keys:
        print("\nUnexpected keys from source model:")
        for key in unexpected_keys:
            print(f"- {key}")
    
    # Move model to device
    model = model.to(device)
    
    # Set up fine-tuning
    optimizer = torch.optim.AdamW(
        [
            # Fine-tune backbone with smaller learning rate
            {"params": model.backbone.parameters(), "lr": lr/10},
            # Train new regression head with normal learning rate
            {"params": model.output_layer.parameters(), "lr": lr}
        ],
        weight_decay=1e-4
    )
    
    # Create checkpoint path
    checkpoint_dir = "./checkpoints/transfer"
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(
        checkpoint_dir, 
        f"ast_{task}_transfer_{feat_type}.pth"
    )
    
    # Fine-tune the model
    print(f"\nFine-tuning for {task} prediction...")
    trainer = Training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=epochs,
        save_path=save_path,
        device=device,
        overfit_batch=False,
        feat_type=feat_type
    )
    
    results = trainer.train_with_eval()

    return model, results


if __name__ == "__main__":
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 1e-5
    
    # Create dataset with multitask targets
    dataset = SpectrogramDataset(
        path="./data/multitask_dataset",
        train=True,
        feat_type='mel',
        max_length=150,
        multitask=True  # This enables returning all three targets
    )
    
    # Create data loaders
    train_loader, val_loader, _ = torch_train_val_split(
        dataset,
        batch_train=BATCH_SIZE,
        batch_eval=BATCH_SIZE
    )
    
    # Initialize model
    backbone = ASTBackbone(
        input_fdim=128,
        input_tdim=150,
        feature_size=100
    )
    model = MultitaskRegressor(backbone).to(DEVICE)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Train using existing train_with_eval function
    save_path = "./checkpoints/multitask_model.pth"
    trainer = Training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=EPOCHS,
        save_path=save_path,
        device=DEVICE,
        overfit_batch=False,
        feat_type='mel'
    )
    
    results = trainer.train_with_eval()


# Transfer Learning
#if __name__ == "__main__":
#    # Configuration
#    SOURCE_MODEL_PATH = "./checkpoints/checkpoint.pth"  # Path to your pre-trained genre model
#    DATASET_PATH = "./data/multitask_dataset"  # Path to sentiment dataset
#    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    
#    # Perform transfer learning
#    model, results = setup_transfer_learning(
#        source_model_path=SOURCE_MODEL_PATH,
#        dataset_path=DATASET_PATH,
#        feat_type='mel',
#        task='valence',
#        device=DEVICE,
#        epochs=20  # Fewer epochs for fine-tuning
#    )


#if __name__ == "__main__":
#    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    print(f"Using device: {DEVICE}")
#    base_path = Path("./data")
#
#    #lstm_paths = [
#    #    base_path / "fma_genre_spectrograms",
#    #    base_path / "fma_genre_spectrograms_beat",
#    #    base_path / "fma_genre_spectrograms",
#    #    base_path / "fma_genre_spectrograms",
#    #]
#    #feat_types = ["mel", "mel", "chroma", "fused"]
#    
#    overfit_batch = False
#
#    #for li, feat_type in zip(lstm_paths, feat_types):
#    #    # Data Params
#    #    dataset_path = li
#    #    batch_size = 32
#    #    max_length = 150
#    #    train_data = True
##
#    #    # Create datasets
#    #    train_dataset = SpectrogramDataset(
#    #        dataset_path,
#    #        class_mapping=CLASS_MAPPING,
#    #        train=train_data,
#    #        max_length=max_length,
#    #        feat_type=feat_type,
#    #    )
##
#    #    # Create data loaders
#    #    train_loader, val_loader, _ = torch_train_val_split(
#    #        train_dataset,
#    #        batch_train=batch_size,
#    #        batch_eval=batch_size,
#    #    )
##
#    #    # Training Hyperparams
#    #    epochs = 100
##
#    #    # Directory to save checkpoints
#    #    checkpoint_dir = "./checkpoints"
#    #    os.makedirs(checkpoint_dir, exist_ok=True)
#    #    CP_PATH = "checkpoint.pth"
#    #    cp_path = os.path.join(checkpoint_dir, CP_PATH)
##
#    #    # LSTM Hyperparams
#    #    input_dim = train_dataset.feat_dim
#    #    dropout = 0.4
#    #    rnn_size = 64
#    #    num_layers = 4
#    #    bidirectional = True
#    #    num_classes = 10
##
#    #    # init LSTM
#    #    backbone = LSTMBackbone(input_dim, rnn_size, num_layers, bidirectional, dropout)
#    #    model = Classifier(num_classes, backbone).to(DEVICE)
##
#    #    # Optimizer
#    #    lr = 1e-4
#    #    weight_decay = 1e-4
#    #    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
##
#    #    # init training
#    #    lstm_ = Training(
#    #        model,
#    #        train_loader,
#    #        val_loader,
#    #        optimizer,
#    #        epochs,
#    #        cp_path,
#    #        DEVICE,
#    #        overfit_batch=overfit_batch,
#    #        feat_type=feat_type + (" beat" if "beat" in str(li) else ""),
#    #    )
#    #    # start training
#    #    lstm_.train_with_eval()
#        
#    dataset_path = "./data/fma_genre_spectrograms"
#    batch_size = 32
#    max_length = 150
#    feat_type = "mel"
#    train_data = True
#    num_classes = 10
#    lr = 1e-4
#    weight_decay = 1e-4
#    epochs = 100
#    checkpoint_dir = "./checkpoints"
#    os.makedirs(checkpoint_dir, exist_ok=True)
#    CP_PATH = "checkpoint.pth"
#    cp_path = os.path.join(checkpoint_dir, CP_PATH)
#    
#    # Create datasets
#    train_dataset = SpectrogramDataset(
#        dataset_path,
#        class_mapping=CLASS_MAPPING,
#        train=train_data,
#        max_length=max_length,
#        feat_type=feat_type,
#    )
#    # Create data loaders
#    train_loader, val_loader, _ = torch_train_val_split(
#        train_dataset,
#        batch_train=batch_size,
#        batch_eval=batch_size,
#    )
#        
#    # CNN Hyperparams
#    lr = 1e-5
#    #input_shape = (train_dataset.max_length, train_dataset.feat_dim)
#    #cnn_in_channels = 1
#    #cnn_filters = [32, 64, 128, 256]
#    #cnn_out_feature_size = 1000
#    ## init CNN
#    #backbone = CNNBackbone(
#    #    input_shape, cnn_in_channels, cnn_filters, cnn_out_feature_size
#    #)
#    #model = Classifier(num_classes, backbone).to(DEVICE)
#    #
#    ## init optimizer
#    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#    ## init training
#    #cnn_ = Training(
#    #    model,
#    #    train_loader,
#    #    val_loader,
#    #    optimizer,
#    #    epochs,
#    #    cp_path,
#    #    DEVICE,
#    #    overfit_batch=overfit_batch,
#    #    feat_type=feat_type,
#    #)
#    ## start training
#    #cnn_.train_with_eval()
#    
#    
#    # AST Hyperparams
#    #input_fdim = 128
#    #input_tdim = 150
#    #feature_size = 100
#    ## init AST
#    #backbone = ASTBackbone(
#    #    input_fdim=input_fdim, input_tdim=input_tdim, 
#    #    feature_size=feature_size,
#    #)
#    #model = Classifier(10, backbone).to(DEVICE)
#    ## init optimizer
#    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#    ## init training
#    #ast_ = Training(
#    #    model,
#    #    train_loader,
#    #    val_loader,
#    #    optimizer,
#    #    epochs,
#    #    cp_path,
#    #    DEVICE,
#    #    overfit_batch=overfit_batch,
#    #    feat_type=feat_type,
#    #)
#    ## start training
#    #ast_.train_with_eval()
#    
#    
#    """
#    
#    
#    
#    
#    """
#    
#    
#    # Main execution for regression tasks
#    #tasks = ['valence', 'energy', 'danceability']
#    #results = {}
#    #dataset_path = "./data/multitask_dataset"
##
#    #for task in tasks:
#    #    print(f"\nTraining regression models for {task}")
#    #    results[task] = setup_regression_training(
#    #        task=task,
#    #        dataset_path=dataset_path,
#    #        feat_type=feat_type,
#    #        max_length=max_length,
#    #        batch_size=batch_size,
#    #        device=DEVICE,
#    #        lr=lr,
#    #        weight_decay=weight_decay,
#    #        epochs=epochs,
#    #        cp_path=cp_path,
#    #        overfit_batch=overfit_batch
#    #    )
#
#
#    
#    # Regressor
#    #tasks = ['valence', 'energy', 'danceability']
#    #test_results = {
#    #    'lstm': [],
#    #    'cnn': [],
#    #    'ast': []
#    #}
#    #
#    #for task in tasks:
#    #    label_index = tasks.index(task) + 1
#    #    train_dataset = SpectrogramDataset(
#    #        dataset_path, class_mapping=CLASS_MAPPING,
#    #        train=True, feat_type=feat_type, max_length=max_length, regression=label_index
#    #    )
#    #    train_loader, val_loader, test_loader = torch_train_val_split(
#    #        train_dataset, batch_train=batch_size, batch_eval=batch_size, test=True
#    #    )
#    #    # Create LSTM Backbone Regressor
#    #    # LSTM Hyperparams
#    #    input_dim = train_dataset.feat_dim
#    #    dropout = 0.0
#    #    rnn_size = 4
#    #    num_layers = 1
#    #    bidirectional = False
#    #    # init LSTM
#    #    backbone = LSTMBackbone(input_dim, rnn_size, num_layers, bidirectional, dropout)
#    #    model = Regressor(backbone).to(DEVICE)
#    #    # init optimizer
#    #    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#    #    # init training
#    #    lstm_regressor = Training(model, train_loader, val_loader, optimizer, epochs, cp_path, DEVICE, overfit_batch=overfit_batch)
#    #    # start training
#    #    lstm_regressor.train_with_eval()
#    #    # Create CNN Backbone Regressor
#    #    # CNN Hyperparams
#    #    input_shape = (train_dataset.max_length, train_dataset.feat_dim)
#    #    cnn_in_channels = 1
#    #    cnn_filters = [32, 64, 128, 256]
#    #    cnn_out_feature_size = 1000
#    #    # init CNN
#    #    backbone = CNNBackbone(input_shape, cnn_in_channels, cnn_filters, cnn_out_feature_size)
#    #    model = Regressor(backbone).to(DEVICE)
#    #    # init optimizer
#    #    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#    #    # init training
#    #    cnn_regressor = Training(model, train_loader, val_loader, optimizer, epochs, cp_path, DEVICE, overfit_batch=overfit_batch)
#    #    cnn_regressor.train_with_eval()
#    #    # Create AST Backbone Regressor
#    #    # AST Hyperparams
#    #    input_fdim = 128
#    #    input_tdim = 150
#    #    feature_size = 100
#    #    # init AST
#    #    backbone = ASTBackbone(input_fdim=input_fdim, input_tdim=input_tdim, feature_size=feature_size)
#    #    model = Regressor(backbone).to(DEVICE)
#    #    # init optimizer
#    #    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#    #    # init training
#    #    ast_regressor = Training(model, train_loader, val_loader, optimizer, epochs, cp_path, DEVICE, overfit_batch=overfit_batch)
#    #    ast_regressor.train_with_eval()

#if __name__ == "__main__":
#    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    print(f"Using device: {DEVICE}")
#
#    # Data Params
#    dataset_path = "./data/fma_genre_spectrograms"
#    batch_size = 32
#    max_length = 150
#    feat_type = "mel"
#    train_data = True
#
#    # Create datasets
#    train_dataset = SpectrogramDataset(
#        dataset_path,
#        class_mapping=CLASS_MAPPING,
#        train=train_data,
#        max_length=max_length,
#        feat_type=feat_type,
#    )
#
#    # Create data loaders
#    train_loader, val_loader, _ = torch_train_val_split(
#        train_dataset,
#        batch_train=batch_size,
#        batch_eval=batch_size,
#    )
#
#    # Training Hyperparams
#    epochs = 100
#    overfit_batch = False
#
#    # Directory to save checkpoints
#    checkpoint_dir = "./checkpoints"
#    os.makedirs(checkpoint_dir, exist_ok=True)
#    CP_PATH = "checkpoint.pth"
#    cp_path = os.path.join(checkpoint_dir, CP_PATH)
#
#    # LSTM Hyperparams
#    input_dim = train_dataset.feat_dim
#    dropout = 0.4
#    rnn_size = 128
#    num_layers = 6
#    bidirectional = True
#    num_classes = 10
#
#    # init LSTM
#    backbone = LSTMBackbone(input_dim, rnn_size, num_layers, bidirectional, dropout)
#    model = Classifier(num_classes, backbone).to(DEVICE)
#
#    # Optimizer
#    lr = 1e-4
#    weight_decay = 1e-4
#    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#    # init training
#    lstm_ = Training(
#        model,
#        train_loader,
#        val_loader,
#        optimizer,
#        epochs,
#        cp_path,
#        DEVICE,
#        overfit_batch=overfit_batch,
#    )
#    # start training
#    lstm_.train_with_eval()
#
#    # CNN Hyperparams
#    input_shape = (train_dataset.max_length, train_dataset.feat_dim)
#    cnn_in_channels = 1
#    cnn_filters = [32, 64, 128, 256]
#    cnn_out_feature_size = 1000
#    # init CNN
#    backbone = CNNBackbone(
#        input_shape, cnn_in_channels, cnn_filters, cnn_out_feature_size
#    )
#    model = Classifier(num_classes, backbone).to(DEVICE)
#    #
#    # init optimizer
#    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#    # init training
#    cnn_ = Training(
#        model,
#        train_loader,
#        val_loader,
#        optimizer,
#        epochs,
#        cp_path,
#        DEVICE,
#        overfit_batch=overfit_batch,
#    )
#    # start training
#    cnn_.train_with_eval()
#
#    # AST Hyperparams
#    input_fdim = 128
#    input_tdim = 150
#    feature_size = 100
#    # init AST
#    backbone = ASTBackbone(
#        input_fdim=input_fdim, input_tdim=input_tdim, feature_size=feature_size
#    )
#    model = Classifier(10, backbone).to(DEVICE)
#
#    # init optimizer
#    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#    # init training
#    ast_ = Training(
#        model,
#        train_loader,
#        val_loader,
#        optimizer,
#        epochs,
#        cp_path,
#        DEVICE,
#        overfit_batch=overfit_batch,
#    )
#    # start training
#    ast_.train_with_eval()


# if __name__ == "__main__":
#    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    print(f"Using device: {DEVICE}")
#
#    # Data Params
#    dataset_path = './data/fma_genre_spectrograms'
#    batch_size = 32
#    max_length = 150
#    feat_type = 'mel'
#    train_data = True
#
#    # Create datasets
#    train_dataset = SpectrogramDataset(
#        dataset_path,
#        class_mapping=CLASS_MAPPING,
#        train=train_data,
#        max_length=max_length,
#        feat_type=feat_type
#    )
#
#    # Create data loaders
#    train_loader, val_loader, _ = torch_train_val_split(
#        train_dataset,
#        batch_train=batch_size,
#        batch_eval=batch_size,
#    )
#
#    # Training Hyperparams
#    epochs = 100
#    overfit_batch = False
#
#    # Directory to save checkpoints
#    checkpoint_dir = './checkpoints'
#    os.makedirs(checkpoint_dir, exist_ok=True)
#    CP_PATH = 'checkpoint.pth'
#    cp_path = os.path.join(checkpoint_dir, CP_PATH)
#
#    # LSTM Hyperparams
#    input_dim = train_dataset.feat_dim
#    dropout = 0.4
#    rnn_size = 128
#    num_layers = 4
#    bidirectional = True
#    num_classes = 10
#
#    # init LSTM
#    backbone = LSTMBackbone(input_dim, rnn_size, num_layers, bidirectional, dropout)
#    model = Classifier(num_classes, backbone).to(DEVICE)
#
#    # Optimizer
#    lr = 1e-4
#    weight_decay = 1e-4
#    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#    # init training
#    lstm_ = Training(model, train_loader, val_loader, optimizer, epochs, cp_path, DEVICE, overfit_batch=overfit_batch)
#    # start training
#    lstm_.train_with_eval()
#
#
#    # CNN Hyperparams
#    input_shape = (train_dataset.max_length, train_dataset.feat_dim)
#    cnn_in_channels = 1
#    cnn_filters = [32, 64, 128, 256]
#    cnn_out_feature_size = 1000
#
#    # init CNN
#    backbone = CNNBackbone(input_shape, cnn_in_channels, cnn_filters, cnn_out_feature_size)
#    model = Classifier(num_classes, backbone).to(DEVICE)
#
#    # init optimizer
#    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#    # init training
#    cnn_ = Training(model, train_loader, val_loader, optimizer, epochs, cp_path, DEVICE, overfit_batch=overfit_batch)
#    # start training
#    cnn_.train_with_eval()
#
#    # AST Hyperparams
#    input_fdim = 128
#    input_tdim = 150
#    feature_size = 100
#
#    # init AST
#    backbone = ASTBackbone(input_fdim=input_fdim, input_tdim=input_tdim, feature_size=feature_size)
#    model = Classifier(10, backbone).to(DEVICE)
#
#    # init optimizer
#    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#    # init training
#    ast_ = Training(model, train_loader, val_loader, optimizer, epochs, cp_path, DEVICE, overfit_batch=overfit_batch)
#    # start training
#    ast_.train_with_eval()
#
#    # Regressor
#
#    tasks = ['valence', 'energy', 'danceability']
#
#    test_results = {
#        'lstm': [],
#        'cnn': [],
#        'ast': []
#    }
#
#    for task in tasks:
#        label_index = tasks.index(task) + 1
#
#        train_dataset = SpectrogramDataset(
#            dataset_path, class_mapping=CLASS_MAPPING,
#            train=True, feat_type=feat_type, max_length=max_length, regression=label_index
#        )
#
#        train_loader, val_loader, test_loader = torch_train_val_split(
#            train_dataset, batch_train=batch_size, batch_eval=batch_size, test=True
#        )
#
#        # Create LSTM Backbone Regressor
#        # LSTM Hyperparams
#        input_dim = train_dataset.feat_dim
#        dropout = 0.0
#        rnn_size = 4
#        num_layers = 1
#        bidirectional = False
#
#        # init LSTM
#        backbone = LSTMBackbone(input_dim, rnn_size, num_layers, bidirectional, dropout)
#        model = Regressor(backbone).to(DEVICE)
#
#        # init optimizer
#        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#        # init training
#        lstm_regressor = Training(model, train_loader, val_loader, optimizer, epochs, cp_path, DEVICE, overfit_batch=overfit_batch)
#        # start training
#        lstm_regressor.train_with_eval()
#
#        # Create CNN Backbone Regressor
#        # CNN Hyperparams
#        input_shape = (train_dataset.max_length, train_dataset.feat_dim)
#        cnn_in_channels = 1
#        cnn_filters = [32, 64, 128, 256]
#        cnn_out_feature_size = 1000
#
#        # init CNN
#        backbone = CNNBackbone(input_shape, cnn_in_channels, cnn_filters, cnn_out_feature_size)
#        model = Regressor(backbone).to(DEVICE)
#
#        # init optimizer
#        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#        # init training
#        cnn_regressor = Training(model, train_loader, val_loader, optimizer, epochs, cp_path, DEVICE, overfit_batch=overfit_batch)
#        cnn_regressor.train_with_eval()
#
#        # Create AST Backbone Regressor
#        # AST Hyperparams
#        input_fdim = 128
#        input_tdim = 150
#        feature_size = 100
#
#        # init AST
#        backbone = ASTBackbone(input_fdim=input_fdim, input_tdim=input_tdim, feature_size=feature_size)
#        model = Regressor(backbone).to(DEVICE)
#
#        # init optimizer
#        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#        # init training
#        ast_regressor = Training(model, train_loader, val_loader, optimizer, epochs, cp_path, DEVICE, overfit_batch=overfit_batch)
#        ast_regressor.train_with_eval()
