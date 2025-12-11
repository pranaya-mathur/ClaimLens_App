"""
Forgery Detection Models
ResNet50-based binary classifier for image forgery detection
"""
import torch.nn as nn
import torchvision.models as models


class ForgeryDetectorCNN(nn.Module):
    """
    ResNet50-based binary classifier for image forgery detection.
    Output: logits over [AUTHENTIC, FORGED].
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
