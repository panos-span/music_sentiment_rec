from convolution import CNNBackbone
from lstm import LSTMBackbone
from modules import Classifier, MultitaskRegressor, Regressor
from sklearn.metrics import classification_report, confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
import numpy as np
from train_utils import validate_one_epoch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
import torch
from dataset import SpectrogramDataset, torch_train_val_split, CLASS_MAPPING



def ensure_numpy(tensor_or_array):
    """
    Safely convert a PyTorch tensor or NumPy array to a NumPy array.
    
    Args:
        tensor_or_array: PyTorch tensor, NumPy array, or Python list
        
    Returns:
        numpy.ndarray: The input converted to a NumPy array
    """
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().numpy()
    elif isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array
    return np.array(tensor_or_array)

def predict(model, test_loader, device):
    """
    Generate predictions for a model on test data.
    
    Args:
        model: Neural network model (Classifier or Regressor)
        test_loader: DataLoader containing test data
        device: Device to run predictions on (cuda/cpu)
        
    Returns:
        tuple: (predictions, true_labels) as Python lists
    """
    model = model.to(device)
    model.eval()
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # Handle different batch formats
            if isinstance(batch, dict):
                inputs = batch['spectrogram'].to(device)
                labels = batch['label']  # Keep labels on CPU
                lengths = batch.get('lengths', None)
            else:
                inputs = batch[0].to(device)
                labels = batch[1]  # Keep labels on CPU
                lengths = batch[2] if len(batch) > 2 else None
            
            # Forward pass
            if lengths is not None:
                outputs = model(inputs, lengths=lengths)
            else:
                outputs = model(inputs)
            
            if isinstance(outputs, tuple):
                outputs = outputs[1]
            
            # Convert predictions based on model type
            if hasattr(model, 'num_classes'):
                predictions = outputs.argmax(dim=1).cpu()
            else:
                predictions = outputs.squeeze().cpu()
            
            # Convert to NumPy and then to Python lists
            predictions = ensure_numpy(predictions)
            labels = ensure_numpy(labels)
            
            # Ensure integer type for classification
            if hasattr(model, 'num_classes'):
                predictions = predictions.astype(int)
                labels = labels.astype(int)
            
            all_predictions.extend(predictions.tolist())
            all_true_labels.extend(labels.tolist())
    
    return all_predictions, all_true_labels

def evaluate(model, test_loader, device, title=None):
    """
    Evaluate model performance with proper type handling for classification and regression.
    
    Args:
        model: Neural network model (Classifier or Regressor)
        test_loader: DataLoader containing test data
        device: Device to run evaluation on (cuda/cpu)
        title: Optional title for plots
        
    Returns:
        dict: Evaluation metrics and results
    """
    # Get predictions and ensure they're Python lists
    y_pred, y_true = predict(model, test_loader, device)
    
    # Handle regression case
    if not hasattr(model, 'num_classes'):
                # Convert predictions and targets to numpy arrays
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        # Check if we're doing multitask regression
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            # Initialize results dictionary for each task
            task_names = ['valence', 'arousal', 'danceability']
            results = {}
            
            # Calculate metrics for each task separately
            for i, task in enumerate(task_names):
                # Get predictions and true values for this task
                task_pred = y_pred[:, i]
                task_true = y_true[:, i]
                
                # Calculate metrics exactly as in single-task case
                spearman_corr, p_value = spearmanr(task_true, task_pred)
                mse = np.mean((task_true - task_pred) ** 2)
                mae = np.mean(np.abs(task_true - task_pred))
                
                # Store metrics for this task
                results[task] = {
                    "spearman_correlation": float(spearman_corr),
                    "spearman_p_value": float(p_value),
                    "mean_squared_error": float(mse),
                    "mean_absolute_error": float(mae),
                    "predictions": task_pred.tolist(),
                    "true_values": task_true.tolist()
                }
                
                # Print metrics for this task
                print(f"\nRegression Metrics for {task}:")
                print(f"Spearman correlation: {spearman_corr:.4f} (p={p_value:.4f})")
                print(f"Mean Squared Error: {mse:.4f}")
                print(f"Mean Absolute Error: {mae:.4f}")
            
            return results
            
        else:
            # Single-task regression case remains unchanged
            y_pred = y_pred.ravel()
            y_true = y_true.ravel()
            
            spearman_corr, p_value = spearmanr(y_true, y_pred)
            mse = np.mean((y_true - y_pred) ** 2)
            mae = np.mean(np.abs(y_true - y_pred))
            
            results = {
                "spearman_correlation": float(spearman_corr),
                "spearman_p_value": float(p_value),
                "mean_squared_error": float(mse),
                "mean_absolute_error": float(mae),
                "predictions": y_pred.tolist(),
                "true_values": y_true.tolist()
            }
            
            print(f"\nRegression Metrics:")
            print(f"Spearman correlation: {spearman_corr:.4f} (p={p_value:.4f})")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Mean Absolute Error: {mae:.4f}")
            
            return results
    
    # Handle classification case
    else:
        # Convert predictions and true labels to Python integers
        y_pred = [int(label) if isinstance(label, (np.number, np.ndarray)) else int(label) 
                 for label in y_pred]
        y_true = [int(label) if isinstance(label, (np.number, np.ndarray)) else int(label) 
                 for label in y_true]
        
        # Get class names with error handling
        num_classes = model.num_classes
        try:
            class_names = [str(test_loader.dataset.label_transformer.inverse(label))
                         for label in range(num_classes)]
        except (AttributeError, KeyError):
            class_names = [str(i) for i in range(num_classes)]
        
        # Calculate accuracy
        accuracy = float(accuracy_score(y_true, y_pred))
        
        # Generate classification report with error handling
        try:
            report = classification_report(
                y_true, y_pred,
                target_names=class_names,
                zero_division=0,
                output_dict=True
            )
            
            # Convert any NumPy types in the report to Python natives
            report = {
                str(k): (
                    {str(k2): float(v2) if isinstance(v2, (np.number, np.ndarray)) else v2
                     for k2, v2 in v.items()}
                    if isinstance(v, dict) else
                    float(v) if isinstance(v, (np.number, np.ndarray)) else v
                )
                for k, v in report.items()
            }
            
        except Exception as e:
            print(f"Warning: Error generating classification report: {e}")
            report = {}
        
        # Generate confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Convert confusion matrix to native Python types
            cm_list = [[float(cell) for cell in row] for row in cm_normalized]
            
            # Plot confusion matrix
            plot_confusion_matrix(
                cm_normalized,
                classes=class_names,
                normalize=True,
                title=title or "Confusion Matrix"
            )
        except Exception as e:
            print(f"Warning: Error generating confusion matrix: {e}")
            cm_list = None
        
        # Prepare results dictionary with only Python native types
        results = {
            "accuracy": float(accuracy),
            "confusion_matrix": cm_list,
            "per_class_metrics": {
                str(class_name): metrics
                for class_name, metrics in report.items()
                if class_name not in ["accuracy", "macro avg", "weighted avg"]
            } if report else {},
            "macro_avg": report.get("macro avg", {}),
            "weighted_avg": report.get("weighted avg", {}),
            "predictions": y_pred,  # Already Python list of ints
            "true_labels": y_true   # Already Python list of ints
        }
        
        # Print metrics
        print(f"\nClassification Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        if report:
            print("\nDetailed Classification Report:")
            print(classification_report(
                y_true, y_pred,
                target_names=class_names,
                zero_division=0
            ))
        
        return results
    


def kaggle_submission(model, test_dataloader, device="cuda"):
    outputs = evaluate(model, test_dataloader, device=device)
    # TODO: Write a csv file for your kaggle submmission
    raise NotImplementedError("You need to implement this")


#if __name__ == "__main__":
    #backbone = LSTMBackbone(
    #    input_dim=128, rnn_size=128, num_layers=2, bidirectional=True, dropout=0.2
    #)
    #model = Classifier(backbone, num_classes=10)
    #test_dataloader, _ = torch_train_val_split(
    #    SpectrogramDataset(
    #        "data/fma_genre_spectrograms",
    #        class_mapping=CLASS_MAPPING,
    #        train=False,
    #        max_length=150,
    #        feat_type="mel",
    #    ),
    #    batch_train=32,
    #    batch_eval=32,
    #)
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #dict = evaluate(model, test_dataloader, device=device)
    #print(dict)
    ## kaggle_submission(model, test_dataloader, device=device)
#
    ## Regressors
#
    #tasks = ["valence", "energy", "danceability"]
