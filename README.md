# WaferWise : Smarter inspection. Faster yields.

## Edge-AI Semiconductor Defect Classification: Explainable edge-AI system for real-time wafer defect detection using lightweight CNN optimized for NXP i.MX RT series deployment.

## Overview

This project implements a production-ready defect classification system for semiconductor wafer inspection with:
- **Lightweight Architecture**: MobileNetV3-Small (1.1M parameters, <5MB quantized)
- **Edge-Optimized**: <100ms inference on NXP i.MX RT1170
- **Explainable AI**: Grad-CAM visualization for defect localization
- **ONNX Export**: Ready for NXP eIQ Toolkit deployment

## Quick Start

### Installation

```bash
git clone <your-repo-url>
cd Edge-AI-Semiconductor-Defect-Classification
pip install -r requirements.txt
```

### Dataset Preparation

Download the [Roboflow Wafer Defect Dataset](https://universe.roboflow.com/ailab-lobb3/wafer-defect-detection) and extract to `data/raw/`:

```bash
python scripts/prepare_roboflow_data.py
```

### Training

```bash
python src/train.py --data_dir data/processed_real --output_dir outputs --num_classes 2 --epochs 30
```

### Evaluation

```bash
python src/evaluate.py --model_path outputs/model_best.pth --data_dir data/processed_real --num_classes 2
```

### Inference with Explainability

```bash
python src/inference.py --image_path path/to/image.jpg --model_path outputs/model_best.pth --num_classes 2
```

### ONNX Export for Edge Deployment

```bash
python src/export_onnx.py --model_path outputs/model_best.pth --output_path outputs/model.onnx --num_classes 2
```

## Project Structure

```
├── src/
│   ├── train.py                    # Training pipeline with augmentation
│   ├── evaluate.py                 # Evaluation with confusion matrix
│   ├── inference.py                # Inference with Grad-CAM visualization
│   ├── export_onnx.py              # ONNX export for edge deployment
│   ├── models/
│   │   └── mobilenet_classifier.py # MobileNetV3-Small architecture
│   ├── data/
│   │   └── dataset.py              # Dataset loader with augmentation
│   └── utils/
│       ├── gradcam.py              # Grad-CAM explainability
│       └── metrics.py              # Evaluation metrics
├── scripts/
│   └── prepare_roboflow_data.py    # Dataset preparation script
└── tests/
    └── test_model.py               # Model unit tests
```

## Model Architecture

- **Backbone**: MobileNetV3-Small (ImageNet pre-trained)
- **Input**: 224×224 grayscale images
- **Classification Head**: 576 → 256 → 128 → num_classes
- **Regularization**: Dropout (0.3, 0.2)
- **Parameters**: 1.1M (trainable)

## Training Configuration

- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **Loss**: Cross-Entropy with label smoothing
- **Scheduler**: ReduceLROnPlateau (patience=5)
- **Early Stopping**: Patience=10
- **Data Augmentation**: 
  - Rotation (±15°)
  - Horizontal/Vertical flip
  - Brightness/Contrast adjustment
  - Gaussian noise

## Performance Metrics

Trained on Roboflow Wafer Defect Dataset (760 images):
- **Validation Accuracy**: 100%
- **Training Accuracy**: 99.79%
- **Model Size**: 4.3MB (FP32), <2MB (INT8 quantized)
- **Classes**: 2 (CMP Scratch, Other)

## Edge Deployment

The model is designed for deployment on NXP i.MX RT series:
1. Export to ONNX format
2. Quantize to INT8 using NXP eIQ Toolkit
3. Deploy on NXP i.MX RT1170 (Cortex-M7 @ 1GHz)
4. Target inference: <100ms per image

## Technology Stack

- **Framework**: PyTorch 2.0+
- **Augmentation**: Albumentations
- **Export**: ONNX
- **Visualization**: OpenCV, Matplotlib
- **Explainability**: Grad-CAM

## Dataset Sources

- [Roboflow Wafer Defect Detection](https://universe.roboflow.com/ailab-lobb3/wafer-defect-detection) - 760 labeled images
- [Roboflow Wafer Classification](https://universe.roboflow.com/waferdetection/wafer-defect-detection-zfi8y) - 126 images
- [IEEE DataPort Wafer Surface](https://ieee-dataport.org/documents/wafer-surface-defect) - 500 annotated images
- [MixedWM38 Wafer Map](https://github.com/Junliangwangdhu/WaferMap) - 38k wafer maps

## Author

Yashasvi Gupta - +91 9316411714
