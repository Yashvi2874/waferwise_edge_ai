"""
Utility functions for metrics and visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate precision, recall, F1-score for each class
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    metrics = {}
    for i, class_name in enumerate(class_names):
        metrics[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': support[i]
        }
    
    return metrics


def plot_training_history(history, output_dir):
    """
    Plot training and validation loss/accuracy curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training history plot saved to {output_dir}/training_history.png")
