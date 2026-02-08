"""
Setup script for wafer defect classification
"""
from setuptools import setup, find_packages

setup(
    name="wafer-defect-classifier",
    version="0.1.0",
    description="Edge-AI Defect Classification for Semiconductor Inspection",
    author="Your Team Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "albumentations>=1.3.0",
    ],
)
