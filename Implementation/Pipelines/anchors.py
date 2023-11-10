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
    

    def create_ROI(self, margin: float): 
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

    
    # TODO: Implement the two new methods for vectorized IoU:
    def create_anchors(self):
        # Create anchor top-left and bottom-right coordinates
        anchor_tl_x = (self.grid_x - self.width / 2).unsqueeze(1)
        anchor_tl_y = (self.grid_y - self.height / 2).unsqueeze(0)
        anchor_br_x = (self.grid_x + self.width / 2).unsqueeze(1)
        anchor_br_y = (self.grid_y + self.height / 2).unsqueeze(0)

        # Repeat coordinates to create a full grid
        anchor_tl_x = anchor_tl_x.repeat(1, len(self.grid_y))
        anchor_tl_y = anchor_tl_y.repeat(len(self.grid_x), 1)
        anchor_br_x = anchor_br_x.repeat(1, len(self.grid_y))
        anchor_br_y = anchor_br_y.repeat(len(self.grid_x), 1)

        return anchor_tl_x, anchor_tl_y, anchor_br_x, anchor_br_y
    
    
    def calculate_iou(self, anchor_tl_x, anchor_tl_y, anchor_br_x, anchor_br_y, box):
        # Convert box coordinates to tensors
        box_tl_x, box_tl_y, box_br_x, box_br_y = box

        # Calculate intersection top-left and bottom-right
        inter_tl_x = torch.max(anchor_tl_x, box_tl_x)
        inter_tl_y = torch.max(anchor_tl_y, box_tl_y)
        inter_br_x = torch.min(anchor_br_x, box_br_x)
        inter_br_y = torch.min(anchor_br_y, box_br_y)

        # Calculate intersection area
        inter_area = (inter_br_x - inter_tl_x).clamp(min=0) * (inter_br_y - inter_tl_y).clamp(min=0)

        # Calculate anchor and box areas
        anchor_area = (anchor_br_x - anchor_tl_x) * (anchor_br_y - anchor_tl_y)
        box_area = (box_br_x - box_tl_x) * (box_br_y - box_tl_y)

        # Calculate union area
        union_area = anchor_area + box_area - inter_area

        # Compute IoU
        iou = inter_area / union_area

        return iou

    
    def get_ROI_indices(self, feature_map, ROIs_list):

        '''In: feature_map, (bs, C, H, W),
            ROIs_list: list(Box2D)
            Out: {ROI #, list((idx_x_min, idx_y_min), (idx_x_max, idx_y_max))}
        '''

        ROIs_indices = {}
        for ROI in ROIs_list:
            # Find the indices in grid_x and grid_y for the given (x_min, y_min) point of anchor:
            idx_x_min = int(torch.searchsorted(self.grid_x, ROI.x_anchor, right=True))
            idx_y_min = int(torch.searchsorted(self.grid_y, ROI.y_anchor, right=True))

            idx_x_max = int(torch.searchsorted(self.grid_x, ROI.x_max, right=True))
            idx_y_max = int(torch.searchsorted(self.grid_y, ROI.y_max, right=True))
            ROIs_indices[ROI] = [(idx_x_min, idx_y_min), (idx_x_max, idx_y_max)] 
        
        return ROIs_indices 