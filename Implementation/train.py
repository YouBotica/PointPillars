import os
import pdb
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import random
import copy
import math

# Pipelines (a.k.a parts of the Neural Network)
from Pipelines.kitti_loader import KITTIDataset
from Pipelines.pillarizer import PillarFeatureNet, Pillarization, PseudoImageDataset
from Pipelines.backbone import BackBone
from Pipelines.detection_head import DetectionHead
from Pipelines.anchors import Box2D, Anchor
from Pipelines.loss import PointPillarLoss
from Pipelines.network import PointPillarsModel

from Utils.transformations import transform_to_canvas, transform_to_grid, map_to_img
from Utils.iou import calculate_iou
from Utils.collate import normalize_annotations
from Utils.boxes import create_boxes_tensor # FIXME: Should be in visualization instead

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


# Create a collate function to handle variable-sized labels:
def collate_batch(batch):
    point_clouds, annotations = zip(*batch)
    point_clouds = torch.stack(point_clouds, dim=0)
    normalized_annotations = normalize_annotations(annotations, pillar_size=PILLAR_SIZE,
        x_lims=(X_MIN, X_MAX), y_lims=(Y_MIN, Y_MAX))
    
    return point_clouds, normalized_annotations


print(f'Can I can use GPU now? -- {torch.cuda.is_available()}')


'''Create data loaders'''
small_train_pointclouds_dir = '/home/adlink/Documents/ECE-57000/ClassProject/Candidate2/PointPillars/dataset/kitti/training/small_train_velodyne'
small_train_labels_dir = '/home/adlink/Documents/ECE-57000/ClassProject/Candidate2/PointPillars/dataset/kitti/training/small_labels_velodyne'



# IMPORTANT: Set to CPU for pillarization otherwise, expect GPU memory to overflow
device =  torch.device('cpu')

train_set = KITTIDataset(pointcloud_dir=small_train_pointclouds_dir, labels_dir=small_train_labels_dir)
        
# Create the dataset and DataLoader
dataset = PseudoImageDataset(pointcloud_dir=small_train_pointclouds_dir, device=device, kitti_dataset=train_set, aug_dim=AUG_DIM, max_points_in_pillar=MAX_POINTS_PER_PILLAR,
                             max_pillars=MAX_FILLED_PILLARS, x_min=X_MIN, y_min=Y_MIN, z_min=Z_MIN, x_max = X_MAX, y_max=Y_MAX,
                             z_max = Z_MAX, pillar_size=PILLAR_SIZE)

train_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)



'''Start training loop'''
n_epochs = 7
model = PointPillarsModel()
loss_fn = PointPillarLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = 0.0

for epoch in range(n_epochs):
    print(f'Epoch: {epoch}')

    # Reset Kitti dataset indexer that retrieves pointclouds and labels:
    train_set = KITTIDataset(pointcloud_dir=small_train_pointclouds_dir, labels_dir=small_train_labels_dir)

    dataset = PseudoImageDataset(pointcloud_dir=small_train_pointclouds_dir, device=device, kitti_dataset=train_set, aug_dim=AUG_DIM, max_points_in_pillar=MAX_POINTS_PER_PILLAR,
                             max_pillars=MAX_FILLED_PILLARS, x_min=X_MIN, y_min=Y_MIN, z_min=Z_MIN, x_max = X_MAX, y_max=Y_MAX,
                             z_max = Z_MAX, pillar_size=PILLAR_SIZE)
    
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_batch)
    

    '''Enable training mode'''
    model.train()

    for batch_idx, (pseudo_images, batched_labels) in enumerate(train_loader):

        gt_boxes_tensor = create_boxes_tensor(batched_labels, attributes_idx)
        
        # Check if gt_boxes_tensor is empty for the current batch
        if gt_boxes_tensor.nelement() == 0:
            print(f'Encountered an empty element on the batch')
            continue
        
        # Get the roi indices:
        roi_indices = anchor.get_ROI_indices(gt_boxes_tensor=gt_boxes_tensor, scale_factor=1.5, 
                    feature_map_size=(H,W))


        # Get IoU tensor and regression targets:
        iou_tensor = anchor.calculate_batch_iou(gt_boxes_tensor) 
        '''IoU tensor (batch_size, n_boxes, num_anchors_x, num_anchors_y)'''

        # Regression targets from ground truth labels
        regression_targets_tensor = anchor.get_regression_targets_tensor(iou_tensor, (H,W), threshold=0.5)

        # Classification targets:
        classification_targets_dict = anchor.get_classification_targets(iou_tensor=iou_tensor, feature_map_size=(H,W),
                                    background_lower_threshold=0.05, background_upper_threshold=0.25)
        
        '''Enable gradients'''
        optimizer.zero_grad()

        loc, size, clf, occupancy, angle, heading = model(pseudo_images)

        
        loss = loss_fn(regression_targets=regression_targets_tensor, classification_targets_dict=classification_targets_dict,
        gt_boxes_tensor = gt_boxes_tensor, loc=loc, size=size, clf=clf, occupancy=occupancy, angle=angle, heading=heading,
        anchor=anchor)

        print(f'Loss: {loss}')
        print(f'Epoch: {epoch}')
        print(f'Batch: {batch_idx}')

        # Backpropagation
        loss.backward()
        optimizer.step()