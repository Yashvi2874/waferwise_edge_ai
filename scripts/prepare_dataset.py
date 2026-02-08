"""
Prepare and organize dataset into train/val/test splits
"""
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict


DEFECT_CLASSES = ['clean', 'crack', 'short', 'open', 'bridge', 'cmp_scratch', 'other']


def create_directory_structure(base_dir):
    """Create train/val/test directory structure"""
    for split in ['train', 'val', 'test']:
        for class_name in DEFECT_CLASSES:
            os.makedirs(os.path.join(base_dir, split, class_name), exist_ok=True)


def split_dataset(image_paths, train_ratio=0.7, val_ratio=0.15):
    """Split dataset into train/val/test"""
    random.shuffle(image_paths)
    
    n = len(image_paths)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return {
        'train': image_paths[:train_end],
        'val': image_paths[train_end:val_end],
        'test': image_paths[val_end:]
    }


def organize_dataset(raw_dir, processed_dir):
    """Organize raw images into processed structure"""
    print("Organizing dataset...")
    
    # Create directory structure
    create_directory_structure(processed_dir)
    
    # Collect images by class
    class_images = defaultdict(list)
    
    # Scan raw directory
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                
                # Infer class from directory name or filename
                class_name = 'other'  # default
                for defect_class in DEFECT_CLASSES:
                    if defect_class in root.lower() or defect_class in file.lower():
                        class_name = defect_class
                        break
                
                class_images[class_name].append(file_path)
    
    # Split and copy files
    stats = defaultdict(lambda: defaultdict(int))
    
    for class_name, images in class_images.items():
        print(f"\nProcessing class: {class_name} ({len(images)} images)")
        
        splits = split_dataset(images)
        
        for split_name, split_images in splits.items():
            for img_path in split_images:
                # Generate new filename
                filename = f"{class_name}_{stats[split_name][class_name]:04d}.png"
                dest_path = os.path.join(
                    processed_dir, split_name, class_name, filename
                )
                
                # Copy file
                shutil.copy2(img_path, dest_path)
                stats[split_name][class_name] += 1
    
    # Print statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    for split_name in ['train', 'val', 'test']:
        print(f"\n{split_name.upper()}:")
        total = 0
        for class_name in DEFECT_CLASSES:
            count = stats[split_name][class_name]
            total += count
            print(f"  {class_name:15s}: {count:4d} images")
        print(f"  {'TOTAL':15s}: {total:4d} images")
    
    print("\n" + "=" * 60)
    print("✓ Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    
    if not os.path.exists(raw_dir):
        print(f"Error: {raw_dir} not found!")
        print("Please run: python scripts/download_data.py")
        exit(1)
    
    organize_dataset(raw_dir, processed_dir)
