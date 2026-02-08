import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from models.mobilenet_classifier import create_model
from data.dataset import create_dataloaders
from utils.metrics import plot_training_history


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc='Training')
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return loss_sum / total, 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Validation'):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            loss_sum += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    
    return loss_sum / total, 100. * correct / total


def train_model(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_loader, val_loader, _ = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    
    print("Creating model...")
    model = create_model(num_classes=args.num_classes, device=device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    patience = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, 'model_best.pth'))
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            patience += 1
        
        if patience >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_final.pth'))
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    plot_training_history(history, args.output_dir)
    
    print(f"\nTraining complete! Best val accuracy: {best_acc:.2f}%")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/processed_real')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(args)
