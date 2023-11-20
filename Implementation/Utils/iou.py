
import torch
import numpy as np
import torch.nn as nn


# Define the IoU function
def calculate_iou(box1, box2):
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1.x_anchor, box2.x_anchor)
    y_top = max(box1.y_anchor, box2.y_anchor)
    x_right = min(box1.x_anchor + box1.width, box2.x_anchor + box2.width)
    y_bottom = min(box1.y_anchor + box1.height, box2.y_anchor + box2.height)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of both bounding boxes
    box1_area = box1.width * box1.height
    box2_area = box2.width * box2.height

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou