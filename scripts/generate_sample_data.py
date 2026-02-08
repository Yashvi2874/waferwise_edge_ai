"""
Generate synthetic sample data for testing (when real data not available)
"""
import os
import cv2
import numpy as np
from pathlib import Path


DEFECT_CLASSES = ['clean', 'crack', 'short', 'open', 'bridge', 'cmp_scratch', 'other']


def generate_clean_wafer(size=224):
    """Generate clean wafer image"""
    img = np.ones((size, size), dtype=np.uint8) * 200
    # Add some noise
    noise = np.random.normal(0, 10, (size, size))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def generate_crack_defect(size=224):
    """Generate wafer with crack defect"""
    img = generate_clean_wafer(size)
    # Draw crack line
    x1, y1 = np.random.randint(0, size//2), np.random.randint(0, size)
    x2, y2 = np.random.randint(size//2, size), np.random.randint(0, size)
    cv2.line(img, (x1, y1), (x2, y2), 50, thickness=2)
    return img


def generate_scratch_defect(size=224):
    """Generate wafer with scratch defect"""
    img = generate_clean_wafer(size)
    # Draw multiple scratches
    for _ in range(3):
        x1 = np.random.randint(0, size)
        y1 = np.random.randint(0, size)
        x2 = x1 + np.random.randint(-50, 50)
        y2 = y1 + np.random.randint(-50, 50)
        cv2.line(img, (x1, y1), (x2, y2), 100, thickness=1)
    return img


def generate_spot_defect(size=224):
    """Generate wafer with spot defects"""
    img = generate_clean_wafer(size)
    # Draw spots
    for _ in range(np.random.randint(3, 8)):
        x = np.random.randint(20, size-20)
        y = np.random.randint(20, size-20)
        radius = np.random.randint(3, 10)
        cv2.circle(img, (x, y), radius, 50, -1)
    return img


def generate_pattern_defect(size=224):
    """Generate wafer with pattern defect"""
    img = generate_clean_wafer(size)
    # Add grid pattern
    for i in range(0, size, 20):
        cv2.line(img, (i, 0), (i, size), 150, 1)
        cv2.line(img, (0, i), (size, i), 150, 1)
    return img


def generate_sample_dataset(output_dir, samples_per_class=20):
    """Generate synthetic dataset for testing"""
    print("Generating synthetic sample dataset...")
    
    generators = {
        'clean': generate_clean_wafer,
        'crack': generate_crack_defect,
        'short': generate_spot_defect,
        'open': generate_spot_defect,
        'bridge': generate_pattern_defect,
        'cmp_scratch': generate_scratch_defect,
        'other': generate_pattern_defect
    }
    
    for split in ['train', 'val', 'test']:
        n_samples = samples_per_class if split == 'train' else samples_per_class // 4
        
        for class_name in DEFECT_CLASSES:
            class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            generator = generators[class_name]
            
            for i in range(n_samples):
                img = generator(224)
                filename = f"{class_name}_{i:04d}.png"
                filepath = os.path.join(class_dir, filename)
                cv2.imwrite(filepath, img)
            
            print(f"  {split}/{class_name}: {n_samples} images")
    
    print(f"\n✓ Sample dataset generated in {output_dir}")
    print("Note: This is synthetic data for testing only!")
    print("For real training, download actual wafer defect datasets.")


if __name__ == "__main__":
    output_dir = "data/processed"
    generate_sample_dataset(output_dir, samples_per_class=20)
