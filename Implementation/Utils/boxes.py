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



# TODO: Get target gt boxes as tensor from batched labels:
def create_boxes_tensor(annotations, attributes_idx):
    '''Creates bl and ur corners representing a 2D Bounding Box on feature map'''
    # attributes_idx is a dictionary that maps attributes to their positions in the annotations tensor
    num_boxes = annotations.size(1)
    # Initialize tensor to hold boxes [x_min, y_min, x_max, y_max]
    boxes = torch.zeros((annotations.size(0), num_boxes, 4))

    for batch_idx in range(annotations.size(0)):
        for box_idx in range(num_boxes):
            # Extract normalized values
            norm_x_center = annotations[batch_idx, box_idx, attributes_idx['norm_x']]
            norm_y_center = annotations[batch_idx, box_idx, attributes_idx['norm_y']]
            norm_w = annotations[batch_idx, box_idx, attributes_idx['norm_w']]
            norm_l = annotations[batch_idx, box_idx, attributes_idx['norm_l']]

            # Compute x_min, y_min, x_max, y_max
            x_min = norm_x_center - norm_w / 2
            y_min = norm_y_center - norm_l / 2
            x_max = norm_x_center + norm_w / 2
            y_max = norm_y_center + norm_l / 2

            # Assign the box coordinates
            boxes[batch_idx, box_idx, :] = torch.tensor([x_min, y_min, x_max, y_max])

    return boxes



