import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from A_softmax_Loss import compute_loss
import torch.nn.functional as F


class SphereFace(nn.Module):
    def __init__(self, num_classes, feature_dim=512, m=4):
        super(SphereFace, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(512, num_classes)
        self.m = m

    def forward(self, x, y=None):
        features = self.feature_extractor(x)
        features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        logits = self.fc(features)
        
        if y is None:
            return logits
        else:
            loss = compute_loss(logits, y, self.m)
            return logits, loss
        
