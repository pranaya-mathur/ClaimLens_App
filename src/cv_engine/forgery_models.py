"""
Forgery Detection Models
ResNet50-based binary classifiers for document forgery detection
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ForgeryDetectorCNN(nn.Module):
    """
    ResNet50-based binary classifier for image forgery detection.
    Output: logits over [AUTHENTIC, FORGED].
    
    NOTE: This is the original generic forgery detector with complex head.
    Use AadhaarForgeryDetectorCNN or PANForgeryDetectorCNN for document-specific detection.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        backbone = models.resnet50(pretrained=pretrained)
        in_features = backbone.fc.in_features

        backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class AadhaarForgeryDetectorCNN(nn.Module):
    """
    ResNet50 for Aadhaar card forgery detection.
    
    Architecture:
    - Input: 3-channel RGB (224x224)
    - Backbone: ResNet50 (pretrained on ImageNet)
    - Head: Simple Linear(2048 → 2)
    - Output: 2 classes [FORGED, AUTHENTIC]
    
    Training Performance:
    - Validation Accuracy: 99.62%
    - Balanced Accuracy: 99.80%
    - AUC: 0.9999
    - Optimal Threshold: 0.5
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        backbone = models.resnet50(pretrained=pretrained)
        in_features = backbone.fc.in_features
        
        # Simple classification head (matches training architecture)
        backbone.fc = nn.Linear(in_features, 2)
        
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class PANForgeryDetectorCNN(nn.Module):
    """
    ResNet50 adapted for 4-channel (RGB + ELA) PAN card forgery detection.
    
    Architecture:
    - Input: 4-channel (RGB + ELA grayscale) (320x320)
    - Backbone: ResNet50 with modified conv1 for 4 channels
    - Head: Linear(2048 → 1)
    - Output: Single logit (BCEWithLogitsLoss compatible)
    
    Weight Initialization:
    - RGB channels (0-2): Pretrained ImageNet weights
    - ELA channel (3): Mean of RGB channels
    
    Training Performance:
    - Accuracy: 99.19% @ threshold 0.5
    - AUC: 0.9996
    - F1 Score: 0.9942 @ threshold 0.49 (optimal)
    - Precision: 95%+ @ threshold 0.48 (precision-oriented)
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        backbone = models.resnet50(pretrained=pretrained)
        
        # Modify first conv layer to accept 4 channels (RGB + ELA)
        original_weights = backbone.conv1.weight.data.clone()
        backbone.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        with torch.no_grad():
            # Copy pretrained RGB weights
            backbone.conv1.weight[:, :3, :, :] = original_weights
            # Initialize ELA channel as mean of RGB
            backbone.conv1.weight[:, 3:4, :, :] = original_weights.mean(
                dim=1, keepdim=True
            )
        
        # Single output logit for binary classification
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)
        
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
