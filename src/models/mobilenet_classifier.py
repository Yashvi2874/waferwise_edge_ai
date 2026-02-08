import torch
import torch.nn as nn
import torchvision.models as models


class WaferDefectClassifier(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, dropout_rate=0.3):
        super().__init__()
        
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Adapt first layer for grayscale images
        orig_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False
        )
        
        if pretrained:
            with torch.no_grad():
                # Average RGB weights for single channel
                self.backbone.features[0][0].weight = nn.Parameter(
                    orig_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Custom classification head
        feat_dim = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.67),
            nn.Linear(128, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_feature_maps(self, x):
        return self.backbone.features(x)


def create_model(num_classes=7, pretrained=True, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = WaferDefectClassifier(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    return model


if __name__ == "__main__":
    model = create_model(num_classes=7, device='cpu')
    x = torch.randn(1, 1, 224, 224)
    y = model(x)
    
    print(f"Model created successfully")
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
