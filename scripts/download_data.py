"""
Script to download and prepare wafer defect datasets
"""
import os
import requests
from pathlib import Path


def download_roboflow_dataset():
    """
    Instructions to download Roboflow datasets
    """
    print("=" * 60)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("\nThis project uses the following datasets:")
    print("\n1. Wafer Defect Detection (Roboflow) - ~760 images")
    print("   URL: https://universe.roboflow.com/ailab-lobb3/wafer-defect-detection")
    print("   Steps:")
    print("   - Visit the URL above")
    print("   - Click 'Download Dataset'")
    print("   - Select format: 'Folder Structure'")
    print("   - Download and extract to: data/raw/roboflow_760/")
    
    print("\n2. Wafer Defect Classification (Roboflow) - ~126 images")
    print("   URL: https://universe.roboflow.com/waferdetection/wafer-defect-detection-zfi8y")
    print("   Steps:")
    print("   - Visit the URL above")
    print("   - Click 'Download Dataset'")
    print("   - Select format: 'Folder Structure'")
    print("   - Download and extract to: data/raw/roboflow_126/")
    
    print("\n3. Wafer Surface Defect (IEEE DataPort) - ~500 images")
    print("   URL: https://ieee-dataport.org/documents/wafer-surface-defect")
    print("   Steps:")
    print("   - Visit the URL above")
    print("   - Create free IEEE account if needed")
    print("   - Download dataset")
    print("   - Extract to: data/raw/ieee_500/")
    
    print("\n4. MixedWM38 Wafer Map Dataset - 38k images")
    print("   URL: https://github.com/Junliangwangdhu/WaferMap")
    print("   Steps:")
    print("   - git clone https://github.com/Junliangwangdhu/WaferMap.git")
    print("   - Copy dataset to: data/raw/mixedwm38/")
    
    print("\n" + "=" * 60)
    print("After downloading, run: python scripts/prepare_dataset.py")
    print("=" * 60)


if __name__ == "__main__":
    # Create directory structure
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed/train", exist_ok=True)
    os.makedirs("data/processed/val", exist_ok=True)
    os.makedirs("data/processed/test", exist_ok=True)
    
    download_roboflow_dataset()
