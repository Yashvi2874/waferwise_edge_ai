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
from data.dataset import create_dataloaders


def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    preds, labels, probs = [], [], []
    
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc='Evaluating'):
            imgs = imgs.to(device)
            outputs = model(imgs)
            prob = torch.softmax(outputs, dim=1)
            _, pred = outputs.max(1)
            
            preds.extend(pred.cpu().numpy())
            labels.extend(lbls.numpy())
            probs.extend(prob.cpu().numpy())
    
    preds = np.array(preds)
    labels = np.array(labels)
    probs = np.array(probs)
    
    acc = 100. * np.sum(preds == labels) / len(labels)
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    cm = confusion_matrix(labels, preds)
    
    return acc, report, cm, probs


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading test dataset...")
    _, _, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Get class names from dataset
    test_dataset = test_loader.dataset
    class_names = test_dataset.class_names
    num_classes = len(class_names)
    
    print(f"Classes: {class_names}")
    
    print("Loading model...")
    model = create_model(num_classes=num_classes, device=device)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("\nEvaluating...")
    acc, report, cm, probs = evaluate_model(model, test_loader, device, class_names)
    
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {acc:.2f}%")
    print(f"{'='*60}\n")
    print("Classification Report:")
    print(report)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {acc:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    plot_confusion_matrix(cm, class_names,
                         os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_dir', default='data/processed_real')
    parser.add_argument('--output_dir', default='outputs/evaluation')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    main(args)
