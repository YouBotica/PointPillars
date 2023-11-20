import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from Pipelines.kitti_loader import HDF5PillarDataset
from Pipelines.network import PointPillarsModel
from Utils.boxes import create_boxes_tensor
from Visualization.visz_bboxes import visualize_batch_bounding_boxes

# Parameters and file paths
AUG_DIM = 9
H = 500
W = 440
NUM_X_PILLARS = int(70.4 / 0.16)
NUM_Y_PILLARS = int(80.0 / 0.16)
val_data_file = '/media/adlink/6a738988-44b7-4696-ba07-3daeb00e5683/kitti_pillars/val_pillar_data.h5'
pretrained_model_path = "/home/adlink/Documents/ECE-57000/ClassProject/github/PointPillars/Implementation/saved_models/best.pth"

# Enable CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the validation dataset
val_dataset = HDF5PillarDataset(val_data_file)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Initialize and load the pretrained model
model = PointPillarsModel(device=device, aug_dim=AUG_DIM)
model.load_state_dict(torch.load(pretrained_model_path)['model_state_dict'])
model.to(device)
model.eval()

# Inference on the validation dataset
with torch.no_grad():
    for batch_idx, (batched_pillars, _, batched_x_indices, batched_y_indices) in enumerate(val_loader):
        batched_pillars, batched_x_indices, batched_y_indices = batched_pillars.to(device), batched_x_indices.to(device), batched_y_indices.to(device)

        # Forward pass through the model
        loc, size, clf, _, _, _ = model(batched_pillars, batched_x_indices, batched_y_indices, NUM_X_PILLARS, NUM_Y_PILLARS)

        # Process and visualize predictions
        # Note: Add any post-processing steps here (like NMS, thresholding, etc.)
        num_detected_objects = torch.count_nonzero(clf > 0.5)  # Example thresholding
        print(f"Batch {batch_idx}: Detected {num_detected_objects} objects")

        # Visualization (Modify as needed)
        # visualize_batch_bounding_boxes(pseudo_images, boxes_tensor, attributes_idx)

print("Inference completed.")