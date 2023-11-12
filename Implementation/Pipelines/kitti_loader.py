
import os
import pdb
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import random


class KITTIDataset(Dataset):
    def __init__(self, pointcloud_dir, labels_dir):
        """
        Args:
            root_dir (string): Directory with all the point clouds.
        """
        self.pointcloud_dir = pointcloud_dir
        self.pointcloud_files = [f for f in os.listdir(self.pointcloud_dir) if
                                  os.path.isfile(os.path.join(self.pointcloud_dir, f))]
        self.labels_dir = labels_dir
        self.labels_files = [f for f in os.listdir(self.labels_dir) if 
                             os.path.isfile(os.path.join(self.labels_dir, f))]
        

    def __len__(self):
        return len(self.pointcloud_files)
    

    def __getitem__(self, idx):
        # Retrieve pointcloud:
        point_cloud_path = os.path.join(self.pointcloud_dir, self.pointcloud_files[idx])
        point_cloud = self.load_point_cloud_from_bin(point_cloud_path)

        # Retrieve label:
        label_file = self.pointcloud_files[idx]
        label_file = label_file[:-3] + 'txt'
        label_path = os.path.join(self.labels_dir, label_file)
        labels = self.parse_kitti_label_file(label_path)
        return point_cloud, labels
    
    
    def load_point_cloud_from_bin(self, bin_path):
        #print(f'File loaded: {bin_path}')
        with open(bin_path, 'rb') as f:
            content = f.read()
            point_cloud = np.frombuffer(content, dtype=np.float32)
            point_cloud = point_cloud.reshape(-1, 4)  # KITTI point clouds are (x, y, z, intensity)
        return torch.from_numpy(point_cloud)
    
    
    def parse_kitti_label_file(self, label_path):

        labels = {'Car': []}  # Initialize a dictionary for 'Car' and more classes that would like to be added
        classes_list = list(labels.keys()) # Get dictionary keys as list

        with open(label_path, 'r') as file:
            for line in file:
                parts = line.split()
                label_type = parts[0]
                if label_type in classes_list:  # Here more classes can be added
                    truncated = float(parts[1])
                    occluded = int(parts[2])
                    alpha = float(parts[3])
                    bbox = [float(x) for x in parts[4:8]]
                    dimensions = [float(x) for x in parts[8:11]]
                    location = [float(x) for x in parts[11:14]]
                    rotation_y = float(parts[14])
                    # Optionally handle the score if it exists.
                    score = float(parts[15]) if len(parts) > 15 else None
                        

                    annotation = {
                        'truncated': truncated,
                        'occluded': occluded,
                        'alpha': alpha,
                        'bbox': bbox,
                        'dimensions': dimensions,
                        'location': location,
                        'rotation_y': rotation_y,
                        'score': score
                    }

                    
                    labels[label_type].append(annotation)       

                else:
                    continue
             


        return labels