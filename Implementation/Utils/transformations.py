
import torch
import numpy as np
import torch.nn as nn


def transform_to_canvas(x_center, y_center, x_min, y_min, radius, pillar_size):
    x_center_img = (x_center - x_min) // pillar_size[0]
    y_center_img = (y_center - y_min) // pillar_size[1]
    radius_img = radius / pillar_size[0] # TODO: This assumes that pillar_size is the same in x and y
    return x_center_img, y_center_img, radius_img


def transform_to_grid(length, width, height, z_center, orientation, pillar_size):
    length_tr = length // pillar_size[0] # Assumes pillar size is equal in both dimensions
    width_tr = width // pillar_size[0]
    return length_tr, width_tr


