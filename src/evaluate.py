"""
Evaluation script for trained model
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models.mobilenet_classifier import create_model
from data.dataset import create_dataloaders, WaferDefectDataset


def evaluate_model(model, dataloader, device, class_names):
    """
    Evaluate model and generate metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate accuracy
    accuracy = 100. * np.sum(all_preds == all_labels) / len(all_labels)
    
    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, report, cm, all_probs


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {output_path}")


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading test dataset...")
    _, _, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    class_names = WaferDefectDataset.DEFECT_CLASSES
    
    # Load model
    print("Loading model...")
    model = create_model(num_classes=len(class_names), device=device)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Evaluate
    print("\nEvaluating model...")
    accuracy, report, cm, probs = evaluate_model(
        model, test_loader, device, class_names
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}\n")
    print("Classification Report:")
    print(report)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save report
    with open(os.path.join(args.output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm, class_names,
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    print(f"\n✓ Evaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                        help='Output directory for evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    main(args)
