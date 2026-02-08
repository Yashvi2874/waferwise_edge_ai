"""
Dataset loader for wafer defect images
"""
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class WaferDefectDataset(Dataset):
    """
    Dataset class for wafer defect images
    """
    
    DEFECT_CLASSES = [
        'clean', 'crack', 'short', 'open', 
        'bridge', 'cmp_scratch', 'other'
    ]
    
    def __init__(self, root_dir, split='train', transform=None, img_size=224):
        """
        Args:
            root_dir: Root directory containing train/val/test folders
            split: 'train', 'val', or 'test'
            transform: Albumentations transform pipeline
            img_size: Target image size
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform
        
        # Build file list
        self.samples = []
        self.labels = []
        
        split_dir = os.path.join(root_dir, split)
        if os.path.exists(split_dir):
            for class_idx, class_name in enumerate(self.DEFECT_CLASSES):
                class_dir = os.path.join(split_dir, class_name)
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append(os.path.join(class_dir, img_name))
                            self.labels.append(class_idx)
        
        print(f"Loaded {len(self.samples)} images for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            # Fallback: create dummy image if loading fails
            image = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # Resize if needed
        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default: normalize and convert to tensor
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).unsqueeze(0)
        
        label = self.labels[idx]
        
        return image, label


def get_transforms(split='train', img_size=224):
    """
    Get augmentation transforms for train/val/test
    
    Args:
        split: 'train', 'val', or 'test'
        img_size: Target image size
    
    Returns:
        Albumentations transform pipeline
    """
    if split == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])


def create_dataloaders(data_dir, batch_size=32, num_workers=4, img_size=224):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Root directory containing processed data
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        img_size: Target image size
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = WaferDefectDataset(
        data_dir, 
        split='train',
        transform=get_transforms('train', img_size),
        img_size=img_size
    )
    
    val_dataset = WaferDefectDataset(
        data_dir,
        split='val',
        transform=get_transforms('val', img_size),
        img_size=img_size
    )
    
    test_dataset = WaferDefectDataset(
        data_dir,
        split='test',
        transform=get_transforms('test', img_size),
        img_size=img_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
