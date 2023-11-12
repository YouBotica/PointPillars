import torch
import numpy as np
import torch.nn as nn
import pdb

from .backbone import BackBone
from .detection_head import DetectionHead


'''Set up the neural network for training'''
class PointPillarsModel(nn.Module):
    def __init__(self):
        super(PointPillarsModel, self).__init__()
        self.backbone = BackBone(in_channels=64, out_channels=64, device=torch.device('cuda'))
        self.detection_head = DetectionHead(in_channels=384, grid_size_x=500, grid_size_y=440, num_anchors=1, 
                num_classes=2, device=torch.device('cuda'))

    def forward(self, x):
        # Forward pass through backbone and detection head
        features = self.backbone(x)
        loc, size, clf, occupancy, angle, heading = self.detection_head(features)
        return loc, size, clf, occupancy, angle, heading