import torch
import torch.nn as nn


class DinoClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Linear(backbone.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)

        return self.classifier(features)
