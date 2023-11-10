import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import random
import copy
import pdb



# Create 2D bounding box Class:
class Box2D():
    def __init__(self, x_lims: tuple, y_lims: tuple, x_anchor: float, y_anchor: float, height: float, width: float):
        super().__init__()    
        self.x_anchor = x_anchor
        self.y_anchor = y_anchor
        self.height = height
        self.width = width
        self.x_max = self.x_anchor + self.width
        self.y_max = self.y_anchor + self.height
        self.x_lims = x_lims
        self.y_lims = y_lims



    def create_marker(self, color='r'):
        return(patches.Rectangle((self.x_anchor, self.y_anchor), width=self.width, 
                height=self.height, linewidth=2, edgecolor=color, facecolor='none'))
    

    def create_ROI(self, margin: float):  # NOTE: Deprecate?
      '''Returns a Box2D type with ROI'''
      ROI_box = copy.copy(self)
      ROI_box.x_anchor = self.x_anchor - margin
      ROI_box.y_anchor = self.y_anchor - margin
      ROI_box.height = self.height + 2*margin
      ROI_box.width = self.width + 2*margin

      # Clamp if ROI is out of the feature_map before returning:
      if (ROI_box.x_anchor < self.x_lims[0]): 
        ROI_box.x_anchor = self.x_lims[0]
      elif (ROI_box.x_anchor + ROI_box.width > self.x_lims[1]):
        ROI_box.x_anchor = self.x_lims[1]
      
      if (ROI_box.y_anchor < self.y_lims[0]):
        ROI_box.y_anchor = self.y_lims[0]
      elif (ROI_box.y_anchor + ROI_box.height > self.y_lims[1]):
        ROI_box.y_anchor = self.y_lims[1]

      return ROI_box
    

# Create anchors class:
class Anchor():
    def __init__(self, width, height, rotations=0.0):
        super().__init__()  
          
        self.width = width
        self.height = height

    def create_anchor_grid(self, H, W): 

        '''In: feature map (bs, C, H, W)
            Returns: (grid_x, grid_y): ()
        '''
        
        self.num_anchors_x = int(torch.round(W / self.width)) 
        self.num_anchors_y = int(torch.round(H / self.height))
        self.grid_x = torch.linspace(0, W, self.num_anchors_x) 
        self.grid_y = torch.linspace(0, H, self.num_anchors_y) 
    
    def generate_ROI_boxes(self, gt_boxes_tensor, scale_factor):
        """
        Expands each ground truth box in gt_boxes_tensor by scale_factor to create ROI boxes.

        Parameters:
        gt_boxes_tensor -- tensor of ground truth boxes, shape (batch_size, num_boxes, 4)
                          Format of each box: (x_min, y_min, x_max, y_max)
        scale_factor -- factor to scale the ground truth boxes

        Returns:
        roi_boxes -- tensor of ROI boxes, shape (batch_size, num_boxes, 4)
        """
        # Calculate centers, widths, and heights of the ground truth boxes
        centers = (gt_boxes_tensor[:, :, :2] + gt_boxes_tensor[:, :, 2:]) / 2
        sizes = gt_boxes_tensor[:, :, 2:] - gt_boxes_tensor[:, :, :2]

        # Scale sizes and recompute min and max coordinates
        scaled_sizes = sizes * scale_factor
        roi_boxes = torch.zeros_like(gt_boxes_tensor)
        roi_boxes[:, :, :2] = centers - scaled_sizes / 2  # x_min, y_min
        roi_boxes[:, :, 2:] = centers + scaled_sizes / 2  # x_max, y_max

        return roi_boxes    

    
    def create_anchors(self): 
        '''Creates anchors as tensor'''
        # Create anchor top-left and bottom-right coordinates
        self.anchor_tl_x = (self.grid_x - self.width / 2).unsqueeze(1)
        self.anchor_tl_y = (self.grid_y - self.height / 2).unsqueeze(0)
        self.anchor_br_x = (self.grid_x + self.width / 2).unsqueeze(1)
        self.anchor_br_y = (self.grid_y + self.height / 2).unsqueeze(0)

        # Repeat coordinates to create a full grid
        self.anchor_tl_x = self.anchor_tl_x.repeat(1, len(self.grid_y))
        self.anchor_tl_y = self.anchor_tl_y.repeat(len(self.grid_x), 1)
        self.anchor_br_x = self.anchor_br_x.repeat(1, len(self.grid_y))
        self.anchor_br_y = self.anchor_br_y.repeat(len(self.grid_x), 1)

        #return anchor_tl_x, anchor_tl_y, anchor_br_x, anchor_br_y
    
    
    def calculate_iou(self, box):
        # Convert box coordinates to tensors
        box_tl_x, box_tl_y, box_br_x, box_br_y = box
        # Calculate intersection top-left and bottom-right
        inter_tl_x = torch.max(self.anchor_tl_x, box_tl_x)
        inter_tl_y = torch.max(self.anchor_tl_y, box_tl_y)
        inter_br_x = torch.min(self.anchor_br_x, box_br_x)
        inter_br_y = torch.min(self.anchor_br_y, box_br_y)

        # Calculate intersection area
        inter_area = (inter_br_x - inter_tl_x).clamp(min=0) * (inter_br_y - inter_tl_y).clamp(min=0)

        # Calculate anchor and box areas
        anchor_area = (self.anchor_br_x - self.anchor_tl_x) * (self.anchor_br_y - self.anchor_tl_y)
        box_area = (box_br_x - box_tl_x) * (box_br_y - box_tl_y)

        # Calculate union area
        union_area = anchor_area + box_area - inter_area

        # Compute IoU
        iou = inter_area / union_area

        return iou
    
    
    def calculate_batch_iou(self, gt_boxes_tensor):
        # Expand anchor coordinates to match batch size and number of boxes
        # Assuming gt_boxes_tensor is of shape (batch_size, n_boxes, 4)
        # and contains [x1, y1, x2, y2] coordinates for the boxes
        batch_size, n_boxes, _ = gt_boxes_tensor.shape
        
        # Reshape anchor coordinates to allow broadcast with gt_boxes_tensor
        # Expanding dimensions to (batch_size, n_boxes, num_anchors_x, num_anchors_y)
        anchor_tl_x = self.anchor_tl_x.expand(batch_size, n_boxes, -1, -1)
        anchor_tl_y = self.anchor_tl_y.expand(batch_size, n_boxes, -1, -1)
        anchor_br_x = self.anchor_br_x.expand(batch_size, n_boxes, -1, -1)
        anchor_br_y = self.anchor_br_y.expand(batch_size, n_boxes, -1, -1)

        # Reshape gt_boxes_tensor to allow broadcast with anchor coordinates
        box_tl_x = gt_boxes_tensor[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        box_tl_y = gt_boxes_tensor[:, :, 1].unsqueeze(-1).unsqueeze(-1)
        box_br_x = gt_boxes_tensor[:, :, 2].unsqueeze(-1).unsqueeze(-1)
        box_br_y = gt_boxes_tensor[:, :, 3].unsqueeze(-1).unsqueeze(-1)

        # Calculate intersections
        inter_tl_x = torch.max(anchor_tl_x, box_tl_x)
        inter_tl_y = torch.max(anchor_tl_y, box_tl_y)
        inter_br_x = torch.min(anchor_br_x, box_br_x)
        inter_br_y = torch.min(anchor_br_y, box_br_y)

        # Calculate intersection and anchor areas
        inter_area = (inter_br_x - inter_tl_x).clamp(min=0) * (inter_br_y - inter_tl_y).clamp(min=0)
        anchor_area = (anchor_br_x - anchor_tl_x) * (anchor_br_y - anchor_tl_y)
        box_area = (box_br_x - box_tl_x) * (box_br_y - box_tl_y)

        # Calculate union and IoU
        union_area = anchor_area + box_area - inter_area
        iou = inter_area / union_area

        # Return the IoU tensor, which will be of shape
        # (batch_size, n_boxes, num_anchors_x, num_anchors_y)
        return iou
    

    def get_regression_targets(self, iou_tensor, feature_map_size, threshold=0.5):
        """
        This method finds the best anchor match for each ground truth box based on the IoU tensor.
        It returns the indices on the feature map grid for anchors with the highest IoU above the threshold.
        If no IoU exceeds the threshold, the indices of the anchor with the highest IoU are selected.

        Parameters:
        iou_tensor -- tensor of IoU values, shape (batch_size, n_boxes, num_anchors_x, num_anchors_y)
        feature_map_size -- size of the feature map grid (H, W)
        threshold -- IoU threshold to consider for positive anchor matching

        Returns:
        A dictionary with keys as batch indices and values as lists of (box_index, feature_map_x_index, feature_map_y_index)
        """
        batch_size, n_boxes, num_anchors_x, num_anchors_y = iou_tensor.shape
        feature_map_h, feature_map_w = feature_map_size
        scale_x = feature_map_w / num_anchors_x
        scale_y = feature_map_h / num_anchors_y
        regression_targets = {batch_idx: [] for batch_idx in range(batch_size)}

        # Iterate through each batch and each ground truth box
        for batch_idx in range(batch_size):
            for box_idx in range(n_boxes):
                # Get the IoU for the current box
                box_iou = iou_tensor[batch_idx, box_idx]

                # Find the max IoU and its index
                max_iou = torch.max(box_iou)
                max_idx = torch.argmax(box_iou)
                anchor_x_idx, anchor_y_idx = np.unravel_index(max_idx.item(), (num_anchors_x, num_anchors_y))

                # Scale anchor indices to feature map indices
                feature_map_x_idx = int(anchor_x_idx * scale_x) # FIXME: Fix offsets
                feature_map_y_idx = int(anchor_y_idx * scale_y)

                if max_iou >= threshold:
                    # If max IoU is above the threshold, use it as the target
                    regression_targets[batch_idx].append((box_idx, feature_map_x_idx, feature_map_y_idx))
                else:
                    # If no IoU exceeds the threshold, take the anchor with the highest IoU
                    regression_targets[batch_idx].append((box_idx, feature_map_x_idx, feature_map_y_idx))

        return regression_targets
    
    
    
    

    def get_ROI_indices(self, roi_boxes, feature_map_size):
        """
        Maps ROI box coordinates to indices on the feature map grid.

        Parameters:
        roi_boxes -- tensor of ROI boxes, shape (batch_size, num_boxes, 4)
                     Format of each box: (x_min, y_min, x_max, y_max)
        feature_map_size -- size of the feature map grid (H, W)

        Returns:
        roi_indices -- dictionary with keys as batch indices and values as lists of tuples (box_index, x_min_idx, y_min_idx, x_max_idx, y_max_idx)
        """
        batch_size, num_boxes, _ = roi_boxes.shape
        feature_map_h, feature_map_w = feature_map_size
        roi_indices = {batch_idx: [] for batch_idx in range(batch_size)}

        for batch_idx in range(batch_size):
            for box_idx in range(num_boxes):
                roi_box = roi_boxes[batch_idx, box_idx]

                # Scale ROI box coordinates to feature map size
                scaled_x_min = int(roi_box[0] * feature_map_w)
                scaled_y_min = int(roi_box[1] * feature_map_h)
                scaled_x_max = int(roi_box[2] * feature_map_w)
                scaled_y_max = int(roi_box[3] * feature_map_h)

                # Clamping to ensure indices are within feature map bounds
                scaled_x_min = max(0, min(scaled_x_min, feature_map_w - 1))
                scaled_y_min = max(0, min(scaled_y_min, feature_map_h - 1))
                scaled_x_max = max(0, min(scaled_x_max, feature_map_w - 1))
                scaled_y_max = max(0, min(scaled_y_max, feature_map_h - 1))

                roi_indices[batch_idx].append((box_idx, scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max))

        return roi_indices


    def plot_regression_targets(self, pseudo_images, regression_targets):
        """
        Plots the regression targets on the feature map.

        Parameters:
        pseudo_images -- feature map tensor, shape (batch_size, C, H, W)
        regression_targets -- dictionary with keys as batch indices and values as lists of (box_index, anchor_x_index, anchor_y_index)
        """
        batch_size, C, H, W = pseudo_images.shape
        # Create a figure with subplots for each batch
        fig, axes = plt.subplots(1, batch_size, figsize=(15, 5 * batch_size))
        if batch_size == 1:
            axes = [axes]  # Make sure axes is iterable
        
        for batch_idx in range(batch_size):
            ax = axes[batch_idx]
            # We will take the first channel of the pseudo image for visualization
            ax.imshow(pseudo_images[batch_idx][0].detach(), cmap='gray')

            # Overlay the regression targets
            if batch_idx in regression_targets:
                for _, x_idx, y_idx in regression_targets[batch_idx]:
                    # Draw a rectangle around the regression target
                    rect = patches.Rectangle(
                        (y_idx - self.width // 2, x_idx - self.height // 2),
                        self.width, self.height,
                        linewidth=1, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)

            # Set the plot titles and labels
            ax.set_title(f'Batch {batch_idx+1} Regression Targets')
            ax.set_xlabel('Feature Map Width')
            ax.set_ylabel('Feature Map Height')

        plt.tight_layout()
        plt.show()
    
    
    def visualize_batch_ious_with_gt_boxes(self, gt_boxes_tensor, iou_tensor):
        # gt_boxes_tensor is of shape (batch_size, n_boxes, 4)
        # and iou_tensor is of shape (batch_size, n_boxes, num_anchors_x, num_anchors_y)
        batch_size = gt_boxes_tensor.shape[0]

        # Create a figure with subplots for each batch
        fig, axes = plt.subplots(1, batch_size, figsize=(15, 5 * batch_size))
        if batch_size == 1:
            axes = [axes]  # Make sure axes is iterable

        for i in range(batch_size):
            ax = axes[i]
            # Visualize the IoU heatmap
            iou_heatmap = torch.max(iou_tensor[i], dim=0).values  # Take the max IoU across all boxes
            ax.imshow(iou_heatmap, cmap='hot', interpolation='nearest')

            # Overlay the ground truth boxes
            for j in range(gt_boxes_tensor.shape[1]):
                box = gt_boxes_tensor[i, j]
                # Draw the box
                rect = patches.Rectangle(
                    (box[0] - self.width / 2, box[1] - self.height / 2),
                    self.width, self.height,
                    linewidth=1, edgecolor='blue', facecolor='none'
                )
                ax.add_patch(rect)

            # Set the plot titles and labels
            ax.set_title(f'Batch {i+1} IoU Heatmap with GT Boxes')
            ax.set_xlabel('Anchor X Coordinate')
            ax.set_ylabel('Anchor Y Coordinate')

        plt.tight_layout()
        plt.show()

    
