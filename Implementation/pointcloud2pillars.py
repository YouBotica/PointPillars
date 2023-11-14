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



# Define a dictionary to map attributes to their indices
attributes_idx = {
    'norm_x': 7,
    'norm_y': 8,
    'norm_z': 9,
    'norm_h': 10,
    'norm_w': 11,
    'norm_l': 12,
}



# Converter from pointcloud to pillars representation:

small_train_pointclouds_dir = '/home/adlink/Documents/ECE-57000/ClassProject/Candidate2/PointPillars/dataset/kitti/training/small_train_velodyne'
small_train_labels_dir = '/home/adlink/Documents/ECE-57000/ClassProject/Candidate2/PointPillars/dataset/kitti/training/small_labels_velodyne'

mini_train_pointclouds_dir = '/home/adlink/Documents/ECE-57000/ClassProject/Candidate2/PointPillars/dataset/kitti/training/mini_train_velodyne'
mini_train_labels_dir = '/home/adlink/Documents/ECE-57000/ClassProject/Candidate2/PointPillars/dataset/kitti/training/mini_label_velodyne'

device =  torch.device('cpu') # CPU should be used for pillarization

train_set = KITTIDataset(pointcloud_dir=small_train_pointclouds_dir, labels_dir=small_train_labels_dir)
pillarizer = Pillarization(device=device, aug_dim=AUG_DIM, x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, 
                                z_min=Z_MIN, z_max=Z_MAX, pillar_size=PILLAR_SIZE, 
                                max_points_per_pillar=MAX_POINTS_PER_PILLAR, max_pillars=MAX_FILLED_PILLARS)



# We'll save the data in an HDF5 file
with h5py.File('/media/adlink/6a738988-44b7-4696-ba07-3daeb00e5683/kitti_pillars/pillar_data.h5', 'w') as h5f:
    # Iterate through all point clouds in the dataset
    for idx in tqdm(range(len(train_set))):
        # Get the point cloud and corresponding label
        point_cloud, label = train_set[idx]

        label_as_tensor = normalize_annotations(annotations=label, pillar_size=PILLAR_SIZE,  # FIXME: ADD A RETURN STATEMENT
                x_lims=(X_MIN, X_MAX), y_lims=(Y_MIN, Y_MAX))
        
        # Pillarize the point cloud
        pillars, x_indices, y_indices = pillarizer.make_pillars(point_cloud)
        
        # Unbatch to store locally:
        pillars = pillars.squeeze(0) 
        x_indices = x_indices.squeeze(0)
        y_indices = y_indices.squeeze(0)
        
        # Convert to numpy and write to HDF5
        grp = h5f.create_group(f'point_cloud_{idx}')
        grp.create_dataset('pillars', data=pillars.numpy())
        grp.create_dataset('x_indices', data=x_indices.numpy())
        grp.create_dataset('y_indices', data=y_indices.numpy())
        grp.create_dataset('label', data=label_as_tensor.numpy()) 