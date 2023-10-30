import torch 
import pdb

class Pillarization:
    def __init__(self, max_points_per_pillar, num_features_per_point, grid_size, pillar_size, min_xyz, max_xyz):
        self.max_points_per_pillar = max_points_per_pillar
        self.num_features_per_point = num_features_per_point
        self.grid_size = grid_size
        self.pillar_size = pillar_size
        self.min_xyz = min_xyz
        self.max_xyz = max_xyz

    def __call__(self, points):
        # Calculate the pillar indices for each point
        pillar_x_indices = ((points[:, 0] - self.min_xyz[0]) / self.pillar_size[0]).floor().long()
        pillar_y_indices = ((points[:, 1] - self.min_xyz[1]) / self.pillar_size[1]).floor().long()


        # Filter out points that are out of range
        mask = (
            (pillar_x_indices >= 0) & 
            (pillar_x_indices < self.grid_size[0]) & 
            (pillar_y_indices >= 0) & 
            (pillar_y_indices < self.grid_size[1])
        )
        pillar_x_indices = pillar_x_indices[mask]
        pillar_y_indices = pillar_y_indices[mask]
        points = points[mask]

        # Create an empty tensor to hold pillar features
        pillars = torch.zeros(
            (self.grid_size[0], self.grid_size[1], self.max_points_per_pillar, self.num_features_per_point),
            dtype=torch.float32
        )
    
# Example Usage:
max_points_per_pillar = 100
num_features_per_point = 4  # x, y, z, intensity
grid_size = (400, 400)  # Number of pillars in x and y direction
pillar_size = (0.2, 0.2)  # Size of each pillar in meters
min_xyz = (-40, -40, -3)
max_xyz = (40, 40, 3)

pillarization = Pillarization(max_points_per_pillar, num_features_per_point, grid_size, pillar_size, min_xyz, max_xyz)


