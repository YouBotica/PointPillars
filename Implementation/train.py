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

# Pipelines (a.k.a parts of the Neural Network)
from Pipelines.kitti_loader import KITTIDataset, HDF5PillarDataset
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
NUM_X_PILLARS = int((X_MAX - X_MIN) / PILLAR_SIZE[0])
NUM_Y_PILLARS= int((Y_MAX - Y_MIN) / PILLAR_SIZE[1])
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

# Enable GPU for training"
device = torch.device('cuda')


'''Create data loaders'''
train_data_file = '/media/adlink/6a738988-44b7-4696-ba07-3daeb00e5683/kitti_pillars/train_data.h5'

train_dataset = HDF5PillarDataset(train_data_file)

# Create train loader as a torch DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


n_epochs = 50
model = PointPillarsModel(device=torch.device('cuda'), aug_dim=AUG_DIM)
loss_fn = PointPillarLoss(feature_map_size=(H, W))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Creates a learning rate scheduler that decays the learning rate every 15 epochs
lr_scheduler = StepLR(optimizer, step_size=15, gamma=0.8)
loss = 0.0

# Define a path to save the model
model_save_path = "/home/adlink/Documents/ECE-57000/ClassProject/github/PointPillars/Implementation/saved_models/fast_model.pth"


for epoch in range(n_epochs):
    model.train()

    for batch_idx, (batched_pillars, batched_labels, batched_x_indices, batched_y_indices) in enumerate(train_loader):
        model.train()

        # Start timer:
        start_time = time.time()

        data_prep_start = time.time()
        gt_boxes_tensor = create_boxes_tensor(batched_labels, attributes_idx)
        
        # Check if gt_boxes_tensor is empty for the current batch
        if gt_boxes_tensor.nelement() == 0:
            print(f'Encountered an empty element on the batch')
            continue


        # Get IoU tensor and regression targets:
        iou_tensor = anchor.calculate_batch_iou(gt_boxes_tensor) 
        '''IoU tensor (batch_size, n_boxes, num_anchors_x, num_anchors_y)'''

        # Regression targets from ground truth labels
        regression_targets_tensor = anchor.get_regression_targets_tensor(iou_tensor, (H,W), threshold=0.5)

        # Classification targets:
        classification_targets_dict = anchor.get_classification_targets(iou_tensor=iou_tensor, feature_map_size=(H,W),
                                    background_lower_threshold=0.05, background_upper_threshold=0.25)
        
        data_prep_end = time.time()


        forward_start = time.time()

        optimizer.zero_grad() # Refresh gradients for forward pass

        loc, size, clf, occupancy, angle, heading, pseudo_images, backbone_out = model(x=batched_pillars, x_orig_indices=batched_x_indices, 
                y_orig_indices=batched_y_indices, num_x_pillars=NUM_X_PILLARS, num_y_pillars=NUM_Y_PILLARS)
        
        forward_end = time.time()
        
        # Loss:
        loss_compute_start = time.time()
        loss = loss_fn(regression_targets=regression_targets_tensor, classification_targets_dict=classification_targets_dict,
        gt_boxes_tensor = gt_boxes_tensor, loc=loc, size=size, clf=clf, occupancy=occupancy, angle=angle, heading=heading,
        anchor=anchor)
        loss_compute_end = time.time()

        # Backpropagation
        backprop_start = time.time()
        loss.backward()
        optimizer.step()
        backprop_end = time.time()

        end_time = time.time()  # End time of the batch
        elapsed_time = end_time - start_time

        
        # Print times and logging:
        print(f'Epoch: {epoch}, batch: {batch_idx}, loss: {loss}, elapsed: {elapsed_time}')
        #print(f'    Data prep time: {data_prep_end - data_prep_start:.2f}s')
        #print(f'    Forward pass time: {forward_end - forward_start:.2f}s')
        #print(f'    Loss computation time: {loss_compute_end - loss_compute_start:.2f}s')
        #print(f'    Backpropagation time: {backprop_end - backprop_start:.2f}s')


        # Save model every 30 batches:
        if batch_idx % 30 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, model_save_path)
            print(f"Model saved to {model_save_path}")

        

        #if epoch % 2 == 0:
        #    continue
        
        # TODO: Add validation here

    lr_scheduler.step()