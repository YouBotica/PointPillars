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



def normalize_annotations(annotations, pillar_size, x_lims, y_lims):
    # Extend num_attributes to store both original and normalized coordinates
    num_attributes = 14  # x, y, z, h, w, l, rotation_y (both original and normalized)
    max_annotations = 20 #len(annotations['Car']) #max(len(a['Car']) for a in annotations) FIXME: Check if this for loop is no longer needed
    
    # Prepare the tensor for all annotations in the batch
    annotations_tensor = torch.zeros(max_annotations, num_attributes)
    

    list_of_gt_objects = annotations['Car']
    for j, car in enumerate(list_of_gt_objects):      
        if car:                          # Original coordinates
            orig_y, orig_z, orig_x = car['location'] # Transformation from camera to velo is applied
            orig_y *= -1 
            orig_z *= -1

            orig_h, orig_w, orig_l = car['dimensions']
            orig_ry = car['rotation_y']

            try:
                # Normalize the location and dimensions to the grid size
                norm_x = (orig_x - x_lims[0]) / pillar_size[0] 
                norm_y = (orig_y - y_lims[0]) / pillar_size[1] 
                norm_z = orig_z / pillar_size[0]  # Assuming Z uses the same pillar size
                norm_h = orig_h / pillar_size[0]  # Assuming H uses the same pillar size
                norm_w = orig_w / pillar_size[1]
                norm_l = orig_l / pillar_size[0]

                # Fill in the tensor with both original and normalized values
                annotations_tensor[j] = torch.tensor([
                    orig_x, orig_y, orig_z, orig_h, orig_w, orig_l, orig_ry,
                    norm_x, norm_y, norm_z, norm_h, norm_w, norm_l, orig_ry  # Note: rotation_y remains the same
                ])
            except:
                print(f'Got index {j} that overflowed the max of {max_annotations}')
                continue

    '''annotations tensor of size (max_gt_boxes, num_attributes) -> (15, 14)'''
    return annotations_tensor



''' DEPRECATED
def normalize_annotations(annotations, pillar_size, x_lims, y_lims):
    # Extend num_attributes to store both original and normalized coordinates
    num_attributes = 14  # x, y, z, h, w, l, rotation_y (both original and normalized)
    max_annotations = len(annotations['Car']) #max(len(a['Car']) for a in annotations) FIXME: Check if this for loop is no longer needed
    
    # Prepare the tensor for all annotations in the batch
    batch_annotations = torch.zeros(len(annotations), max_annotations, num_attributes)
    
    for i, annotation in enumerate(annotations): # Annotations is a list of dictionaries where the elements are vehicle ground truths
        list_of_gt_objects = annotations[annotation]
        for j, car in enumerate(list_of_gt_objects):
            if car:                          # Original coordinates
                orig_y, orig_z, orig_x = car['location'] # Transformation from camera to velo is applied
                orig_y *= -1 
                orig_z *= -1

                orig_h, orig_w, orig_l = car['dimensions']
                orig_ry = car['rotation_y']

                
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
                
        return batch_annotations'''