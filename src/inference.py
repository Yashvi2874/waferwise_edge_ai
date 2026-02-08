"""
Inference script for single image prediction with Grad-CAM
"""
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.mobilenet_classifier import create_model
from utils.gradcam import GradCAM
from data.dataset import WaferDefectDataset


def preprocess_image(image_path, img_size=224):
    """Load and preprocess image"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize
    image_resized = cv2.resize(image, (img_size, img_size))
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_normalized = (image_normalized - 0.5) / 0.5
    
    # To tensor
    image_tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0)
    
    return image_tensor, image_resized


def predict_with_gradcam(model, image_tensor, image_original, device):
    """Run inference with Grad-CAM visualization"""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = probs.max(1)
    
    # Generate Grad-CAM
    target_layer = model.backbone.features[-1]
    gradcam = GradCAM(model, target_layer)
    
    heatmap = gradcam.generate_cam(image_tensor, predicted.item())
    overlay = gradcam.overlay_heatmap(image_original, heatmap)
    
    return predicted.item(), confidence.item(), heatmap, overlay


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading model...")
    model = create_model(num_classes=7, device=device)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Load and preprocess image
    print(f"Processing image: {args.image_path}")
    image_tensor, image_original = preprocess_image(args.image_path)
    
    # Predict
    predicted_class, confidence, heatmap, overlay = predict_with_gradcam(
        model, image_tensor, image_original, device
    )
    
    class_names = WaferDefectDataset.DEFECT_CLASSES
    predicted_label = class_names[predicted_class]
    
    # Display results
    print(f"\nPrediction: {predicted_label}")
    print(f"Confidence: {confidence*100:.2f}%")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Prediction: {predicted_label}\nConfidence: {confidence*100:.1f}%')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if args.output_path:
        plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to {args.output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on single image')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save visualization')
    
    args = parser.parse_args()
    main(args)
