
import os
import pdb
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import pdb



class PillarFeatureNet(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(PillarFeatureNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.device = device
        
        self.to(self.device)

    
    def forward(self, x):
        # Input x is of shape (D, P, N)
        # Convert it to (P, D, N) for 1x1 convolution      
        x = x.to(self.device)
        x = x.permute(1, 0, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # Max pooling operation over the points' dimension
        x, _ = torch.max(x, dim=2)  # Output shape: (P, C)
        return x.T  # Output shape: (C, P)



class Pillarization:
    def __init__(self, device, x_min, x_max, y_min, y_max, z_min, z_max, pillar_size,
                max_points_per_pillar, aug_dim, max_pillars):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.pillar_size = pillar_size
        self.max_points_per_pillar = max_points_per_pillar
        self.aug_dim = aug_dim
        self.num_x_pillars = int((self.x_max - self.x_min) / self.pillar_size[0])
        self.num_y_pillars = int((self.y_max - self.y_min) / self.pillar_size[1])
        self.max_pillars = max_pillars
        self.device = device
        

    def make_pillars(self, points):
        """
        Convert point cloud (x, y, z) into pillars.
        """
        # Mask points outside of our defined boundaries
        
        mask = (
            (points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max) &
            (points[:, 1] >= self.y_min) & (points[:, 1] <= self.y_max) &
            (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
        )
        points = points[mask]
        

        pillars = torch.zeros((self.num_x_pillars, self.num_y_pillars, self.max_points_per_pillar, 
            self.aug_dim)) # Size 
        
        # Count how many points are in each pillar to ensure we don't exceed `max_points_per_pillar`
        count = torch.zeros((self.num_x_pillars, self.num_y_pillars), dtype=torch.long)


        # Send to CUDA if available:
        if (self.device != torch.device('cpu')): # To CUDA if available (GPU?)
            self.x_indices.to(self.device)
            self.y_indices.to(self.device)
            points = points.to(self.device)
            pillars = pillars.to(self.device)
            count = count.to(self.device)

        
        # Calculate x,y indices for each point:
        self.x_indices = points[:,0] - self.x_min / self.pillar_size[0]
        self.y_indices = points[:,1] - self.y_min / self.pillar_size[1] 

        self.x_indices = self.x_indices.int()
        self.y_indices = self.y_indices.int()

   
        # TODO: Store points in the pillars in a vectorized way filling the pillars tensor (Note: Pretty sure there is a faster approach for this)     
        for i in range(points.shape[0]):
            x_ind = self.x_indices[i]
            y_ind = self.y_indices[i]
            
            if count[x_ind, y_ind] < self.max_points_per_pillar:
                # Compute x_c, y_c and z_c
                x_c = (x_ind * self.pillar_size[0] + self.pillar_size[0] / 2.0) - points[i, 0]
                y_c = (y_ind * self.pillar_size[1] + self.pillar_size[1] / 2.0) - points[i, 1]
                z_c = (self.z_min + self.z_max) / 2 - points[i, 2] # assuming the z-center is the midpoint
                
                # Calculate pillar center
                x_pillar_center = (x_ind * self.pillar_size[0] + self.pillar_size[0] / 2.0)
                y_pillar_center = (y_ind * self.pillar_size[1] + self.pillar_size[1] / 2.0)
                
                # Add original x, y, and z coordinates, then x_c, y_c, z_c
                pillars[x_ind, y_ind, count[x_ind, y_ind], :3] = points[i, :3]
                
                if (self.device != torch.device('cpu')): 
                    pillars[x_ind, y_ind, count[x_ind, y_ind], 3:6] = torch.tensor([x_c, y_c, z_c]).to(self.device)
                else: 
                    pillars[x_ind, y_ind, count[x_ind, y_ind], 3:6] = torch.tensor([x_c, y_c, z_c])
                    
                pillars[x_ind, y_ind, count[x_ind, y_ind], 6] = x_pillar_center - pillars[x_ind, y_ind, count[x_ind, y_ind], 0]
                pillars[x_ind, y_ind, count[x_ind, y_ind], 7] = y_pillar_center - pillars[x_ind, y_ind, count[x_ind, y_ind], 1]
                
                count[x_ind, y_ind] += 1
                
        
        # Limit the number of pillars:

        # Flatten the count tensor and get the indices of the top P pillars
        flat_count = count.view(-1)  # Flatten the count tensor to 1D
        _, top_pillar_flat_indices = flat_count.topk(self.max_pillars, largest=True)  # Get the indices of the top pillars
        
        # Convert the flat indices back to 2D indices
        top_pillar_x_indices = (top_pillar_flat_indices // self.num_y_pillars)
        top_pillar_y_indices = (top_pillar_flat_indices % self.num_y_pillars)
        
        # Create a new tensor to hold the top pillars
        top_pillars = torch.zeros((self.max_pillars, self.max_points_per_pillar, self.aug_dim)) # Size (P, N, D)
        
        # Fill in the top pillars
        for i, (x_ind, y_ind) in enumerate(zip(top_pillar_x_indices, top_pillar_y_indices)):
            current_pillar_points = count[x_ind, y_ind]
    
            # Zero-padding if too few point, random sampling if too many points:
            if current_pillar_points > self.max_points_per_pillar:
                # If there are more points than the max, randomly select max_points_per_pillar
                perm = torch.randperm(current_pillar_points)[:self.max_points_per_pillar]
                top_pillars[i] = pillars[x_ind, y_ind][perm]
            elif current_pillar_points < self.max_points_per_pillar:
                # If there are fewer, zero-pad the pillar
                top_pillars[i, :current_pillar_points] = pillars[x_ind, y_ind][:current_pillar_points]
                # Remaining positions are already zero because of the zero initialization of top_pillars                

        # Reshape pillars to size (D,P,N) from (P,N,D):
        top_pillars = top_pillars.permute(2, 0, 1) 

        return top_pillars, top_pillar_x_indices, top_pillar_y_indices
    
    
    
class PseudoImageDataset(Dataset):
    def __init__(self, pointcloud_dir, kitti_dataset, aug_dim, max_points_in_pillar, max_pillars, x_min, x_max, y_min, 
                    y_max, z_min, z_max, pillar_size, device='cpu', transform=None):
        
        # Members of class:
        self.device = device
        self.kitti_dataset = kitti_dataset
        self.aug_dim = aug_dim
        self.max_points_in_pillar = max_points_in_pillar
        self.max_pillars = max_pillars
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.pillar_size = pillar_size # Tuple
        self.pointcloud_dir = pointcloud_dir
        self.transform = transform
        
        # Get pointcloud filenames in specified directory: 
        self.filenames = [f for f in os.listdir(self.pointcloud_dir) if os.path.isfile(os.path.join(self.pointcloud_dir, f))]
        
        
        self.pillarizer = Pillarization(device=self.device, aug_dim=self.aug_dim, x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=40.0, 
                                        z_min=self.z_min, z_max=self.z_max, pillar_size=self.pillar_size, 
                                        max_points_per_pillar=self.max_points_in_pillar, max_pillars=self.max_pillars)

        self.feature_extractor = PillarFeatureNet(in_channels=self.aug_dim, out_channels=64, device=self.device) # Size (C,P) 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        point_cloud, label = self.kitti_dataset[idx] #.load_point_cloud_from_bin(os.path.join(self.pointcloud_dir, self.filenames[idx]))
        pillars, x_orig_indices, y_orig_indices = self.pillarizer.make_pillars(point_cloud)
        
        # Apply linear activation, batchnorm, and ReLU for feature extraction from pillars tensor
        features = self.feature_extractor(pillars) # Output of size (C,P)
        
        # Generate pseudo-image:
        pseudo_image = torch.zeros(features.shape[0], self.pillarizer.num_y_pillars, self.pillarizer.num_x_pillars).to(self.device)
        
        # Scatter the features back to their original pillar locations

        print(f'Loading point cloud number {idx}')

        # Fill pseudo_image with features:
        for c in range(features.shape[0]):
            pseudo_image[c, y_orig_indices, x_orig_indices] = features[c]


        if self.transform:
            pseudo_image = self.transform(pseudo_image)
            
        return pseudo_image, label
