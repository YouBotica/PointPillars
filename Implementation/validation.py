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
from Utils.boxes import create_boxes_tensor # TODO: Should be in visualization instead

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
val_data_file = '/media/adlink/6a738988-44b7-4696-ba07-3daeb00e5683/kitti_pillars/val_pillar_data.h5'


train_dataset = HDF5PillarDataset(train_data_file)

val_dataset = HDF5PillarDataset(val_data_file)

# Create train loader as a torch DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)

# Initialize your model without weights
model = PointPillarsModel(device=torch.device('cuda'), aug_dim=AUG_DIM)

# Define the path to the pretrained model checkpoint
pretrained_model_path = "/home/adlink/Documents/ECE-57000/ClassProject/github/PointPillars/Implementation/saved_models/best.pth"  

# Load the pretrained model checkpoint
checkpoint = torch.load(pretrained_model_path)

# Load the pretrained model's state dictionary into your model
model.load_state_dict(checkpoint['model_state_dict'])


loss_fn = PointPillarLoss(feature_map_size=(H, W))
loss = 0.0


# TODO: Create an mAP function: 
def get_mAP_metric(gt_boxes_tensor, predicted_target_locations, clf):
    '''
    gt_boxes_tensor -- size (bs, max_gt_boxes, 4) where the 4 is (x1, y1, x2, y2)
    predicted_targets -- size (bs, n_boxes, 2) where the 2 is (x_idx, y_idx)
    clf -- size (bs, n_anchors, n_classes+1, H, W)
    '''

    # TODO: Compare each predicted_object's class 

    # Get predicted class:
    x_idxs = predicted_target_locations[:, : ,0].long() # 0 is x, 1 is y
    y_idxs = predicted_target_locations[:, : ,1].long()
    
    # Batch indices
    bs, n_anchors, n_classes_plus_1, H, W = clf.size()
    batch_indices = torch.arange(bs).view(-1, 1, 1, 1, 1) # To match clf dimensions

    # Use indexing to get the predicted class scores at the specified indices
    predicted_scores = clf[batch_indices, :, 1, y_idxs, x_idxs]





'''Validation with unseen data'''

model.eval()
print(f'WARNING: Entering evaluation mode')

start_time = time.time()
with torch.no_grad():
    for batch_idx_val, (batched_pillars_val, batched_labels_val, batched_x_indices_val, batched_y_indices_val) in enumerate(val_loader):

        gt_boxes_tensor_val = create_boxes_tensor(batched_labels_val, attributes_idx)

        # Check if gt_boxes_tensor is empty for the current batch
        if gt_boxes_tensor_val.nelement() == 0:
            print(f'Encountered an empty element on the batch')
            continue


        # Get IoU tensor and regression targets:
        iou_tensor_val = anchor.calculate_batch_iou(gt_boxes_tensor_val) 
        '''IoU tensor (batch_size, n_boxes, num_anchors_x, num_anchors_y)'''

        # Regression targets from ground truth labels
        regression_targets_tensor_val = anchor.get_regression_targets_tensor(iou_tensor_val, (H,W), threshold=0.5)

        # Classification targets:
        classification_targets_dict_val = anchor.get_classification_targets(iou_tensor=iou_tensor_val, feature_map_size=(H,W),
                                    background_lower_threshold=0.05, background_upper_threshold=0.25)
        

        loc_val, size_val, clf_val, occupancy_val, angle_val, heading_val, pseudo_images, backbone_out = model(
            x=batched_pillars_val, x_orig_indices=batched_x_indices_val,  
            y_orig_indices=batched_y_indices_val, num_x_pillars=NUM_X_PILLARS, num_y_pillars=NUM_Y_PILLARS)
        

        loss_val = loss_fn(regression_targets=regression_targets_tensor_val, 
                           classification_targets_dict=classification_targets_dict_val,
                            gt_boxes_tensor = gt_boxes_tensor_val, loc=loc_val, size=size_val, 
                            clf=clf_val, occupancy=occupancy_val, angle=angle_val, heading=heading_val, 
                            anchor=anchor)
        

        metric = get_mAP_metric(gt_boxes_tensor_val, regression_targets_tensor_val, clf_val) 
        

        print(f'Validating with batch {batch_idx_val}, got loss: {loss_val}')


