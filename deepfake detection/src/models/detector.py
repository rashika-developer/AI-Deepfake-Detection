import torch
import torch.nn as nn
from torchvision import models

class DeepfakeDetector(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True, dropout_rate=0.5):
        super(DeepfakeDetector, self).__init__()
        if model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_ftrs, 1)
            )
        elif model_name == 'resnet50':
            self.model = models.resnet50(weights='DEFAULT' if pretrained else None)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_ftrs, 1)
            )
        elif model_name == 'vit':
            self.model = models.vit_b_16(weights='DEFAULT' if pretrained else None)
            num_ftrs = self.model.heads.head.in_features
            self.model.heads.head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_ftrs, 1)
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x

if __name__ == "__main__":
    model = DeepfakeDetector()
    print(model)
