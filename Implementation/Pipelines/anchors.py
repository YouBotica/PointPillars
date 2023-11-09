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

    def create_anchor_grid(self, feature_map): 

        '''In: feature map (bs, C, H, W)
            Returns: (grid_x, grid_y): ()
        '''
        
        self.num_anchors_x = int(torch.round(feature_map.size()[-1] / self.width)) 
        self.num_anchors_y = int(torch.round(feature_map.size()[-2] / self.height))
        self.grid_x = torch.linspace(0, feature_map.size()[-1], self.num_anchors_x) 
        self.grid_y = torch.linspace(0, feature_map.size()[-2], self.num_anchors_y) 
    
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