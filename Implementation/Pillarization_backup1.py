# Pillarization pipeline:
class Pillarization:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, pillar_size, max_points_per_pillar, aug_dim):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.pillar_size = pillar_size
        self.max_points_per_pillar = max_points_per_pillar
        self.aug_dim = aug_dim
    
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
        
        # Create pillar grid
        x_indices = ((points[:, 0] - self.x_min) // self.pillar_size[0]).type(torch.int64)
        y_indices = ((points[:, 1] - self.y_min) // self.pillar_size[1]).type(torch.int64)

        # Calculate the number of pillars in x and y directions
        num_x_pillars = int((self.x_max - self.x_min) / self.pillar_size[0])
        num_y_pillars = int((self.y_max - self.y_min) / self.pillar_size[1])
        

        # Create a tensor to hold the pillars data
        #pillars = torch.zeros((num_x_pillars, num_y_pillars, self.max_points_per_pillar, points.shape[1]))
        pillars = torch.zeros((num_x_pillars, num_y_pillars, self.max_points_per_pillar, self.aug_dim))
        indices_dict = {}
        
        # Fill the pillars:
        for i in range(points.shape[0]):
            x_ind, y_ind = x_indices[i], y_indices[i]
            if pillars[x_ind, y_ind].shape[0] < self.max_points_per_pillar:
                # Calculate mean coordinates of the points inside this pillar
                x_mean = torch.mean(points[x_indices == x_ind, 0])
                y_mean = torch.mean(points[y_indices == y_ind, 1])
                
                # Calculate pillar x-y center:
                pillar_x_center = x_ind * self.pillar_size[0] + self.pillar_size[0] / 2.0
                pillar_y_center = y_ind * self.pillar_size[1] + self.pillar_size[1] / 2.0
                
                # Augment the point with its offset from the pillar's center
                augmented_point = torch.cat((points[i, :3], points[i, 3:], 
                 points[i, :3] - torch.tensor([x_mean, y_mean, 0]),
                 points[i, :2] - torch.tensor([pillar_x_center, pillar_y_center])))
                
                # Insert the augmented point into the pillar tensor
                indices_dict[x_ind, y_ind] = i 
                pillars[x_ind, y_ind, pillars[x_ind, y_ind].shape[0]] = augmented_point
                
                
        # Random sampling or zero padding
        for i in range(num_x_pillars):
            for j in range(num_y_pillars):
                if pillars[i, j].shape[0] > self.max_points_per_pillar:
                    # Randomly sample points if there are too many for a given pillar
                    pillars[i, j] = pillars[i, j][torch.randperm(pillars[i, j].shape[0])[:self.max_points_per_pillar]]
                elif pillars[i, j].shape[0] < self.max_points_per_pillar:
                    # Zero pad if there are too few points for a given pillar
                    pillars[i, j] = torch.cat((pillars[i, j], torch.zeros((self.max_points_per_pillar - pillars[i, j].shape[0], pillars[i, j].shape[1]))))
               
        return pillars
        
    def reshape_pillars(pillars):
        '''Pillars are now of size (num_x_pillars, num_y_pillars, max_points_per_pillar, aug_dim), but
        need to be converted to (D,P,N) where D is aug_dim, P is number of non-empty pillars and N is 
        max points per pillar'''

        # Flatten the x, y dimensions of the pillar tensor to get all pillars in a single dimension
        flattened_pillars = pillars.view(-1, pillars.shape[2], pillars.shape[3])

        # Compute a mask to identify non-empty pillars
        # We can define a non-empty pillar as one that contains at least one non-zero point
        non_empty_mask = (flattened_pillars.sum(dim=[1, 2]) != 0)

        # Use the mask to extract non-empty pillars
        non_empty_pillars = flattened_pillars[non_empty_mask]

        return resized_non_empty_pillars
        
        

        
# Pillarization testing:

# TODO: Get a random sample:
def load_point_cloud_from_bin(bin_path):
    with open(bin_path, 'rb') as f:
        content = f.read()
        point_cloud = np.frombuffer(content, dtype=np.float32)
        point_cloud = point_cloud.reshape(-1, 4)  # KITTI point clouds are (x, y, z, intensity)
    
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]
    
    return point_cloud

# Example usage
random_sample = random.randint(1, 7400)
bin_path = os.path.join(train_dir, '000038.bin')
point_cloud = load_point_cloud_from_bin(bin_path)

# TODO: Pillarize the sample:
D = 9 # Augmented dimension
N = 100 # Max number of points per pillar

pillarizer = Pillarization(aug_dim=D, x_min=0, x_max=70.4, y_min=-40.8, y_max=40.8, 
           z_min=-3, z_max=3, pillar_size=(0.2, 0.2), max_points_per_pillar=N)

pillars = pillarizer.make_pillars(torch.from_numpy(point_cloud))
        
