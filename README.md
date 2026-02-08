# Edge-AI Semiconductor Defect Classification

Real-time wafer defect detection using lightweight CNN optimized for edge deployment on NXP i.MX RT series.

## Features

- **Lightweight Model**: MobileNetV3-Small (1.1M parameters)
- **Edge-Optimized**: <5MB quantized, <100ms inference
- **Explainable AI**: Grad-CAM visualization
- **7 Defect Classes**: Clean, Crack, Short, Open, Bridge, CMP Scratch, Other
- **ONNX Export**: Ready for NXP eIQ deployment

## Quick Start

### Installation

```bash
git clone <your-repo-url>
cd Edge-AI-Semiconductor-Defect-Classification
pip install -r requirements.txt
```

### Test Model

```bash
python src/models/mobilenet_classifier.py
```

### Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

### Train Model

```bash
python src/train.py --data_dir data/processed --output_dir outputs --epochs 30
```

### Evaluate

```bash
python src/evaluate.py --model_path outputs/model_best.pth --data_dir data/processed
```

### Inference with Grad-CAM

```bash
python src/inference.py --image_path path/to/image.png --model_path outputs/model_best.pth
```

### Export to ONNX

```bash
python src/export_onnx.py --model_path outputs/model_best.pth --output_path outputs/model.onnx
```

## Dataset Sources

- [Roboflow Wafer Defect Detection](https://universe.roboflow.com/ailab-lobb3/wafer-defect-detection) - 760 images
- [Roboflow Wafer Classification](https://universe.roboflow.com/waferdetection/wafer-defect-detection-zfi8y) - 126 images
- [IEEE DataPort Wafer Surface](https://ieee-dataport.org/documents/wafer-surface-defect) - 500 images
- [MixedWM38 Wafer Map](https://github.com/Junliangwangdhu/WaferMap) - 38k images

Download datasets and run:
```bash
python scripts/download_data.py  # Instructions
python scripts/prepare_dataset.py  # Organize data
```

## Project Structure

```
├── src/
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Model evaluation
│   ├── inference.py          # Inference + Grad-CAM
│   ├── export_onnx.py        # ONNX export
│   ├── models/
│   │   └── mobilenet_classifier.py
│   ├── data/
│   │   └── dataset.py
│   └── utils/
│       ├── gradcam.py
│       └── metrics.py
├── scripts/
│   ├── download_data.py
│   ├── prepare_dataset.py
│   └── generate_sample_data.py
├── tests/
│   └── test_model.py
└── notebooks/
    └── demo.ipynb
```

## Model Architecture

- **Backbone**: MobileNetV3-Small (ImageNet pre-trained)
- **Input**: 224×224 grayscale images
- **Output**: 7-class softmax
- **Classification Head**: 576 → 256 → 128 → 7
- **Regularization**: Dropout (0.3, 0.2)

## Training Configuration

- **Optimizer**: Adam (lr=1e-3)
- **Loss**: Cross-Entropy
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Patience=10
- **Augmentation**: Rotation, flip, brightness, contrast

## Target Performance

- **Accuracy**: ≥90%
- **Inference**: <100ms on NXP i.MX RT1170
- **Model Size**: <5MB (INT8 quantized)
- **F1-Score**: ≥0.85 for critical defects

## Technology Stack

- PyTorch
- Albumentations
- ONNX
- OpenCV
- Grad-CAM