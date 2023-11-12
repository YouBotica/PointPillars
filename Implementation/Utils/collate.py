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




def normalize_annotations(annotations, pillar_size, x_lims, y_lims):
    # Extend num_attributes to store both original and normalized coordinates
    num_attributes = 14  # x, y, z, h, w, l, rotation_y (both original and normalized)
    max_annotations = max(len(a['Car']) for a in annotations)
    
    # Prepare the tensor for all annotations in the batch
    batch_annotations = torch.zeros(len(annotations), max_annotations, num_attributes)
    
    for i, a in enumerate(annotations):
        for j, car in enumerate(a['Car']):
            if car:  # Check if the annotation is not empty
                # Original coordinates
                orig_y, orig_z, orig_x = car['location'] # Transformation from camera to velo is applied
                orig_h, orig_w, orig_l = car['dimensions']
                orig_ry = car['rotation_y']
                orig_y *= -1
                orig_z *= -1
                
                # Normalize the location and dimensions to the grid size
                norm_x = (orig_x - x_lims[0]) / pillar_size[0] 
                norm_y = (orig_y - y_lims[0]) / pillar_size[1] 
                norm_z = orig_z / pillar_size[0]  # Assuming Z uses the same pillar size
                norm_h = orig_h / pillar_size[0]  # Assuming H uses the same pillar size
                norm_w = orig_w / pillar_size[1]
                norm_l = orig_l / pillar_size[0]
                
                # Fill in the tensor with both original and normalized values
                batch_annotations[i, j] = torch.tensor([
                    orig_x, orig_y, orig_z, orig_h, orig_w, orig_l, orig_ry,
                    norm_x, norm_y, norm_z, norm_h, norm_w, norm_l, orig_ry  # Note: rotation_y remains the same
                ])
    
    return batch_annotations
