import os
import pdb
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import pdb


class DetectionHead(nn.Module):
    def __init__(self, in_channels, grid_size_x, grid_size_y, num_anchors, num_classes, device):
        super(DetectionHead, self).__init__()
        self.device = device
        self.to(self.device) # Send model to GPU
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Assuming 1 anchor boxes per cell
        self.loc_layer = nn.Conv2d(in_channels, num_anchors * 3, 1, device=device)
        self.size_layer = nn.Conv2d(in_channels, num_anchors * 3, 1, device=device)
        self.clf_layer = nn.Conv2d(in_channels, num_anchors * (num_classes + 1), 1, device=device) # +1 for confidence score
        self.occupancy_layer = nn.Conv2d(in_channels, num_anchors * 1, 1, device=device)
        self.angle_layer = nn.Conv2d(in_channels, num_anchors * 1, 1, device=device)
        self.heading_layer = nn.Conv2d(in_channels, num_anchors * 1, 1, device=device)


    def forward(self, x):
        x.to(self.device)
        loc = self.loc_layer(x).view(x.size(0), self.num_anchors, 3, self.grid_size_x, self.grid_size_y)
        size = self.size_layer(x).view(x.size(0), self.num_anchors, 3, self.grid_size_x, self.grid_size_y)

        clf = torch.sigmoid(self.clf_layer(x)).view(x.size(0), 
                self.num_anchors, self.num_classes + 1, self.grid_size_x, self.grid_size_y)
  
        occupancy = self.occupancy_layer(x).view(x.size(0), self.num_anchors, 1, self.grid_size_x, self.grid_size_y)
        angle = self.angle_layer(x).view(x.size(0), self.num_anchors, 1, self.grid_size_x, self.grid_size_y)
        heading = self.heading_layer(x).view(x.size(0), self.num_anchors, 1, self.grid_size_x, self.grid_size_y)

        return loc, size, clf, occupancy, angle, heading
