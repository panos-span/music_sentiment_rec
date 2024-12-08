from convolution import CNNBackbone
from lstm import LSTMBackbone
from modules import Classifier, MultitaskRegressor, Regressor
from sklearn.metrics import classification_report, confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
import numpy as np
from train_utils import validate_one_epoch
from scipy.stats import spearmanr
import torch
from dataset import SpectrogramDataset, torch_train_val_split, CLASS_MAPPING


def predict(model, test_loader, device="cuda"):
    """Predict the labels of a test set using a model"""
    all_y_pred = torch.empty(0).to(device)
    all_y_true = torch.empty(0).to(device)
    for i, (x, y, lengths) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        lengths = lengths.to(device)
        with torch.no_grad():
            y_pred = model(x, y, lengths)
            all_y_pred = torch.cat((all_y_pred, y_pred), dim=0)
            all_y_true = torch.cat((all_y_true, y), dim=0)
            
    return all_y_pred, all_y_true
                
    

def evaluate(model, test_loader, device="cuda", regression=False):
    if regression:
        # Return spearman correlation from predict
        y_pred, y_true = predict(model, test_loader, device)
        return spearmanr(y_pred, y_true)
        
        
    """Evaluate a single model on a test set"""
    # Get unique class names
    num_classes = len(np.unique(test_loader.dataset.labels))
    class_names = [test_loader.dataset.label_transformer.inverse(label) for label in range(num_classes)]
    
    #_ , y_pred , y_true = validate_one_epoch(model, test_loader, device)

    all_y_pred, all_y_true = predict(model, test_loader, device)
    
    # Move tensors to CPU and ensure integers
    y_pred = all_y_pred.cpu().numpy().ravel()
    y_true = all_y_true.cpu().numpy().ravel()
    
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
    print(f"\nResults:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print("\nPer-class metrics:")
    print(classification_report(y_true, y_pred, target_names=class_names,zero_division=0))
    
    cm = confusion_matrix(y_pred, y_true, normalize='true')
    plot_confusion_matrix(cm, classes=test_loader.dataset.labels, normalize=True, title=f'Validation Confusion Matrix')

    
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

    
    
    # return ...  # Return the model predictions


def kaggle_submission(model, test_dataloader, device="cuda"):
    outputs = evaluate(model, test_dataloader, device=device)
    # TODO: Write a csv file for your kaggle submmission
    raise NotImplementedError("You need to implement this")


if __name__ == "__main__":
    backbone = LSTMBackbone(input_dim=128, rnn_size=128, num_layers=2, bidirectional=True, dropout=0.2)
    model = Classifier(backbone, num_classes=10)
    test_dataloader, _ = torch_train_val_split(
        SpectrogramDataset(
            "data/fma_genre_spectrograms",
            class_mapping=CLASS_MAPPING,
            train=False,
            max_length=150,
            feat_type="mel",
        ),
        batch_train=32,
        batch_eval=32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dict = evaluate(model, test_dataloader, device=device)
    print(dict)
    #kaggle_submission(model, test_dataloader, device=device)
    
    # Regressors
    
    tasks = ['valence', 'energy', 'danceability']
    

        
