import torch
import numpy as np
import torch.nn as nn
import pdb

from .backbone import BackBone
from .detection_head import DetectionHead
from .pillarizer import PillarFeatureNet


'''Set up the neural network for training'''
class PointPillarsModel(nn.Module):
    def __init__(self, device, aug_dim):
        super(PointPillarsModel, self).__init__()

        self.device = device
        self.aug_dim = aug_dim
        self.feature_extractor = PillarFeatureNet(in_channels=self.aug_dim, out_channels=64, device=self.device) 
        
        self.backbone = BackBone(in_channels=64, out_channels=64, device=torch.device('cuda'))
        self.detection_head = DetectionHead(in_channels=384, grid_size_x=500, grid_size_y=440, num_anchors=1, 
                num_classes=2, device=torch.device('cuda'))
        

    def forward(self, x, x_orig_indices, y_orig_indices, num_x_pillars, num_y_pillars):   
        '''Forward pass through pillar feature net, backbone and detection head: '''
        '''
        Inputs: 
        x_orig_indices of size (bs, P)
        y_orig_indices of size (bs, P)
        '''
        self.x_orig_indices = x_orig_indices.to(self.device)
        self.y_orig_indices = y_orig_indices.to(self.device)

        # Apply linear activation, batchnorm, and ReLU for feature extraction from pillars tensor
        features = self.feature_extractor(x) # Size (bs, C, P)
        bs, num_channels, num_pillars = features.size()

        # Generate pseudo-image:
        pseudo_images = torch.zeros(bs, num_channels, num_y_pillars, num_x_pillars).to(self.device) # (bs, C, num_y, num_x)


        for b in range(bs):
            # Get the indices for the current batch
            cur_x_indices = self.x_orig_indices[b].long()  
            cur_y_indices = self.y_orig_indices[b].long()  

            # Get the features for the current batch
            cur_features = features[b]  # Size (C, P)

            # The indices must be in the range [0, num_pillars-1], so we might need to clamp them
            cur_x_indices = torch.clamp(cur_x_indices, 0, num_x_pillars - 1)
            cur_y_indices = torch.clamp(cur_y_indices, 0, num_y_pillars - 1)

            # Expand the indices to match the features dimensions
            cur_x_indices = cur_x_indices.unsqueeze(0).expand(num_channels, -1)
            cur_y_indices = cur_y_indices.unsqueeze(0).expand(num_channels, -1)

            # Scatter the features to the correct locations in the pseudo-image
            for c in range(num_channels):
                # We use the index_put_ function here which is the in-place version of index_put
                # This allows us to put cur_features at the specific indices in the pseudo image
                pseudo_images[b][c] = pseudo_images[b][c].index_put_((cur_y_indices[c], cur_x_indices[c]), cur_features[c], accumulate=False)

        #pdb.set_trace()
        #if self.transform: DEPRECATED
        #    pseudo_image = self.transform(pseudo_image)
        backbone_out = self.backbone(pseudo_images)
        loc, size, clf, occupancy, angle, heading = self.detection_head(backbone_out)

        return loc, size, clf, occupancy, angle, heading, pseudo_images, backbone_out