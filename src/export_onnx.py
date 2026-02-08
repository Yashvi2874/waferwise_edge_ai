"""
Export trained model to ONNX format for edge deployment
"""
import argparse
import torch
import onnx
import onnxruntime as ort
import numpy as np

from models.mobilenet_classifier import create_model


def export_to_onnx(model, output_path, img_size=224):
    """Export PyTorch model to ONNX"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, img_size, img_size)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Model exported to {output_path}")


def verify_onnx_model(onnx_path, pytorch_model, img_size=224):
    """Verify ONNX model produces same output as PyTorch"""
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    # Create test input
    test_input = torch.randn(1, 1, img_size, img_size)
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # ONNX inference
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]
    
    # Compare
    diff = np.abs(pytorch_output - onnx_output).max()
    print(f"Max difference between PyTorch and ONNX: {diff:.6f}")
    
    if diff < 1e-5:
        print("✓ ONNX model verified successfully!")
    else:
        print("⚠ Warning: Significant difference detected")
    
    return diff < 1e-5


def main(args):
    device = torch.device('cpu')  # Export on CPU
    
    # Load model
    print("Loading PyTorch model...")
    model = create_model(num_classes=args.num_classes, device=device)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Export to ONNX
    print("Exporting to ONNX...")
    export_to_onnx(model, args.output_path, args.img_size)
    
    # Verify
    if args.verify:
        print("\nVerifying ONNX model...")
        verify_onnx_model(args.output_path, model, args.img_size)
    
    # Print model info
    onnx_model = onnx.load(args.output_path)
    print(f"\nONNX Model Info:")
    print(f"  IR Version: {onnx_model.ir_version}")
    print(f"  Opset Version: {onnx_model.opset_import[0].version}")
    print(f"  Producer: {onnx_model.producer_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained PyTorch model')
    parser.add_argument('--output_path', type=str, default='outputs/model.onnx',
                        help='Output path for ONNX model')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='Number of classes')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify ONNX model output')
    
    args = parser.parse_args()
    main(args)
