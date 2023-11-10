import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn
from Pipelines.anchors import Anchor



def visualize_batch_bounding_boxes(feature_maps, boxes_tensor, attributes_idx, visz_anchor, anchor):
    ''' boxes tensor: orig_x, orig_y, orig_z, orig_h, orig_w, orig_l, orig_ry,
    norm_x, norm_y, norm_z, norm_h, norm_w, norm_l, orig_ry '''

    # Get the batch size and the number of channels
    batch_size, channels, height, width = feature_maps.shape
    
    # Create a figure with subplots
    fig, axes = plt.subplots(batch_size, 1, figsize=(10, batch_size * 5))
    
    for batch_idx in range(batch_size):
        # If there's only one batch, axes might not be an array
        if batch_size == 1:
            ax = axes
        else:
            ax = axes[batch_idx]
        
        if visz_anchor:
            for x in anchor.grid_x: 
                ax.axvline(x=x, color='b', alpha=0.5)

            for y in anchor.grid_y: 
                ax.axhline(y=y, color='b', alpha=0.5)
        
        # Sum over the channel dimension and convert to numpy for visualization
        img = feature_maps[batch_idx].sum(dim=0).detach().cpu().numpy()
        
        # Normalize the image for better contrast
        img = (img - img.min()) / (img.max() - img.min())
        
        # Show the image
        ax.imshow(img, cmap='gray')
        
        # Plot each bounding box for the current item
        for box_idx in range(boxes_tensor.size(1)):
            # Get the bounding box coordinates
            x_min, y_min, x_max, y_max = boxes_tensor[batch_idx, box_idx, :]
            
            # Calculate width and height of the bounding box
            box_w = x_max - x_min
            box_h = y_max - y_min
            
            # Create a rectangle patch
            bbox = patches.Rectangle((x_min, y_min), box_w, box_h, linewidth=2, edgecolor='r', facecolor='none')
            
            # Add the bounding box to the plot
            ax.add_patch(bbox)
        
        # Set the title for the subplot
        ax.set_title(f'Batch {batch_idx + 1}')
    
    # Show the plots
    plt.tight_layout()
    plt.show()