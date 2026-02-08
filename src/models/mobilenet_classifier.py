"""
MobileNetV3-Small based defect classifier for edge deployment
"""
import torch
import torch.nn as nn
import torchvision.models as models


class WaferDefectClassifier(nn.Module):
    """
    Lightweight CNN for wafer defect classification optimized for edge devices.
    Based on MobileNetV3-Small architecture.
    """
    
    def __init__(self, num_classes=7, pretrained=True, dropout_rate=0.3):
        super(WaferDefectClassifier, self).__init__()
        
        # Load pretrained MobileNetV3-Small
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Modify first conv layer for grayscale input (1 channel instead of 3)
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # If pretrained, average the RGB weights for grayscale
        if pretrained:
            with torch.no_grad():
                self.backbone.features[0][0].weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Get feature dimension from backbone
        feature_dim = self.backbone.classifier[0].in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.67),
            nn.Linear(128, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def get_feature_maps(self, x):
        """Extract feature maps for Grad-CAM visualization"""
        features = self.backbone.features(x)
        return features


def create_model(num_classes=7, pretrained=True, device='cuda'):
    """
    Factory function to create model instance
    
    Args:
        num_classes: Number of defect classes
        pretrained: Use ImageNet pretrained weights
        device: Device to load model on
    
    Returns:
        model: WaferDefectClassifier instance
    """
    model = WaferDefectClassifier(
        num_classes=num_classes,
        pretrained=pretrained
    )
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model(num_classes=7, device='cpu')
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 224, 224)
    output = model(dummy_input)
    
    print(f"Model created successfully!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
