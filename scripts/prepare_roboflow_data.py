"""
Prepare Roboflow COCO dataset for classification
"""
import os
import json
import shutil
from collections import defaultdict

# Map Roboflow classes to our classes
CLASS_MAPPING = {
    'crack-particles': 'crack',
    'crease': 'other',
    'scratches': 'cmp_scratch'
}

def prepare_roboflow_data():
    """Convert Roboflow detection dataset to classification format"""
    
    raw_dir = "data/raw"
    processed_dir = "data/processed_real"
    
    # Create directories
    for split in ['train', 'val', 'test']:
        for class_name in ['crack', 'cmp_scratch', 'other']:
            os.makedirs(os.path.join(processed_dir, split, class_name), exist_ok=True)
    
    stats = defaultdict(lambda: defaultdict(int))
    
    # Process train and valid splits
    for split_name in ['train', 'valid']:
        anno_file = os.path.join(raw_dir, split_name, '_annotations.coco.json')
        
        if not os.path.exists(anno_file):
            continue
        
        # Load annotations
        with open(anno_file, 'r') as f:
            data = json.load(f)
        
        # Create category mapping
        categories = {c['id']: c['name'] for c in data['categories']}
        
        # Group images by category
        image_categories = defaultdict(set)
        for ann in data['annotations']:
            img_id = ann['image_id']
            cat_name = categories[ann['category_id']]
            image_categories[img_id].add(cat_name)
        
        # Copy images
        for img_info in data['images']:
            img_id = img_info['id']
            img_name = img_info['file_name']
            
            # Get primary category (first one if multiple)
            if img_id in image_categories:
                orig_class = list(image_categories[img_id])[0]
                mapped_class = CLASS_MAPPING.get(orig_class, 'other')
                
                # Determine output split (use valid as val, split train into train/test)
                if split_name == 'valid':
                    out_split = 'val'
                else:
                    # Use 90% for train, 10% for test (every 10th image goes to test)
                    current_count = stats['train'][mapped_class] + stats['test'][mapped_class]
                    out_split = 'test' if current_count % 10 == 0 else 'train'
                
                # Copy file
                src = os.path.join(raw_dir, split_name, img_name)
                dst = os.path.join(processed_dir, out_split, mapped_class, img_name)
                
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    stats[out_split][mapped_class] += 1
    
    # Print statistics
    print("\n" + "=" * 60)
    print("REAL DATASET STATISTICS")
    print("=" * 60)
    
    for split_name in ['train', 'val', 'test']:
        print(f"\n{split_name.upper()}:")
        total = 0
        for class_name in ['crack', 'cmp_scratch', 'other']:
            count = stats[split_name][class_name]
            total += count
            print(f"  {class_name:15s}: {count:4d} images")
        print(f"  {'TOTAL':15s}: {total:4d} images")
    
    print("\n" + "=" * 60)
    print("✓ Real dataset preparation complete!")
    print("=" * 60)

if __name__ == "__main__":
    prepare_roboflow_data()
