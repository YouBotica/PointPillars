import os
import pdb
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

import time
import random
import copy
import math
import ipdb
import h5py
from tqdm import tqdm

# Pipelines (a.k.a parts of the Neural Network)
from Pipelines.kitti_loader import KITTIDataset
from Pipelines.pillarizer import PillarFeatureNet, Pillarization, PseudoImageDataset
from Pipelines.anchors import Box2D, Anchor


from Utils.transformations import transform_to_canvas, transform_to_grid, map_to_img
from Utils.collate import normalize_annotations 
from Utils.boxes import create_boxes_tensor 

# Visualization tools:
from Visualization.visz_pointcloud_w_label import plot_point_cloud_with_bboxes_o3d
from Visualization.visz_bboxes import visualize_batch_bounding_boxes

# Some Neural Network Parameters:
AUG_DIM = 9
MAX_POINTS_PER_PILLAR = 100
MAX_FILLED_PILLARS = 12000
X_MIN = 0.0
X_MAX = 70.4
Y_MIN = -40.0
Y_MAX = 40.0
Z_MIN = -3.0
Z_MAX = 1.0
PILLAR_SIZE = (0.16, 0.16)
DESIRED_CLASSES = ['Car'] # More classes can be added here
SCALE_FACTOR = 1.5
H = 500
W = 440


ANCHORS = torch.tensor([[3.9, 1.6, 1.56, -1, 0], # Anchors as tensor: (height, width, height, z_center, orientation)
                       [1.6, 3.9, 1.56, -1, 1.5708],
                       [0.8, 0.6, 1.73, -0.6, 0],
                       [0.6, 0.8, 1.73, -0.6, 1.5708]]
                       )

mapped_anchors = ANCHORS.detach().clone()
mapped_anchors[:,0:2] /= PILLAR_SIZE[0]


# Define a dictionary to map attributes to their indices
attributes_idx = {
    'norm_x': 7,
    'norm_y': 8,
    'norm_z': 9,
    'norm_h': 10,
    'norm_w': 11,
    'norm_l': 12,
}

# Create an anchor: 
anchor = Anchor(width=mapped_anchors[0][1], height=mapped_anchors[0][1]) # TODO: Add more anchors for better learning
anchor.create_anchor_grid(H,W) # Creates grid
anchor.create_anchors()


print(f'Can I can use GPU now? -- {torch.cuda.is_available()}')