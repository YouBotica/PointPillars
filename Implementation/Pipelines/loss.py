import os
import pdb
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import pdb



class PointPillarLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, beta_loc = 2.0, beta_cls = 1.0, feature_map_size=(500,440)):
        super(PointPillarLoss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.alpha = alpha
        self.gamma = gamma
        self.beta_cls = beta_cls
        self.beta_loc = beta_loc
        self.feature_map_size = feature_map_size



    def forward(self, regression_targets, classification_targets_dict, 
                gt_boxes_tensor, loc, size, clf, occupancy, angle, heading, anchor):
        
        '''
        Inputs: 
        loc -- size (batch_size, n_anchors, 3, H, W)
        size -- size (batch_size, n_anchors, 3, H, W) 
        clf -- size (batch_size, n_anchors, 3, H, W)
        regression_targets -- tensor of size (batch_size, n_boxes, 2) with the indices of the best matching anchors
        gt_boxes_tensor -- size (bs, n_boxes, 4)
        '''

        da = torch.sqrt(anchor.width**2 + anchor.height**2)

        # Initialize the predictions
        batch_size, n_boxes = regression_targets.shape[:2]
        x_pred = torch.zeros(batch_size, n_boxes, dtype=loc.dtype)
        y_pred = torch.zeros(batch_size, n_boxes, dtype=loc.dtype)
        dx_tensor = torch.zeros(batch_size, n_boxes, dtype=loc.dtype)
        dy_tensor = torch.zeros(batch_size, n_boxes, dtype=loc.dtype)
        dw_tensor = torch.zeros(batch_size, n_boxes, dtype=loc.dtype)
        dl_tensor = torch.zeros(batch_size, n_boxes, dtype=loc.dtype)

     
        # Classification loss:
        '''background probs -- dict{batch: prob_loss}'''
        background_focal_loss = 0.0
        n_classification_target = 0

        for b in range(batch_size):
            for n_target, cls_target in enumerate(classification_targets_dict[b]):
                x_idx = classification_targets_dict[b][n_target][1] #(n_box, x, y)
                y_idx = classification_targets_dict[b][n_target][2]
                '''clf -- size (batch_size, n_anchors, 3, H, W)'''
                clf_val = clf[b][0][0][y_idx][x_idx]
                # Apply focal loss
                background_focal_loss += -torch.log(clf_val)*self.alpha*(1 - clf_val)**self.gamma
                n_classification_target += 1

        if batch_size*n_boxes != 0.0:
            background_focal_loss /= n_classification_target
        else:
            background_focal_loss = 0.0
            print(f'Division by zero encountered on background!')   


        # Regression loss:
        car_focal_loss = 0.0
        for b in range(batch_size):
            for n in range(n_boxes):

                x_idx = regression_targets[b, n, 0].long()  # Ensure the indices are long type
                y_idx = regression_targets[b, n, 1].long()  # Ensure the indices are long type

                if (x_idx >= self.feature_map_size[1] or y_idx >= self.feature_map_size[0]): 
                    print(f'Exceeded an index, x_idx: {x_idx} and boundary: {self.feature_map_size[1]} or y_idx: {y_idx} and boundary: {self.feature_map_size[0]}')
                    continue

                x_pred[b, n] = loc[b, 0, 0, y_idx, x_idx]  # Indexing y first as it corresponds to H dimension
                y_pred[b, n] = loc[b, 0, 1, y_idx, x_idx]  # Indexing y first as it corresponds to H dimension
                w_gt = gt_boxes_tensor[b, n, 3] - gt_boxes_tensor[b, n, 1]
                l_gt = gt_boxes_tensor[b, n, 2] - gt_boxes_tensor[b, n, 0]
                x_gt = gt_boxes_tensor[b, n, 0] + w_gt/2
                y_gt = gt_boxes_tensor[b, n, 1] - l_gt/2
                dx_tensor[b, n] = (x_gt - x_pred[b,n]) / da 
                dy_tensor[b, n] = (y_gt - y_pred[b,n]) / da 

                # Sizes:
                if (w_gt != 0.0):
                    dw_tensor[b, n] = torch.log((w_gt / torch.abs(size[b, 0, 0, y_idx, x_idx])))
                if (l_gt != 0.0):
                    dl_tensor[b, n] = torch.log((l_gt / torch.abs(size[b, 0, 1, y_idx, x_idx])))

                # Classification loss for cars:
                car_prob = clf[b, 0, 1, y_idx, x_idx]
                car_focal_loss += -torch.log(car_prob)*self.alpha*(1 - car_prob)**self.gamma


        if batch_size*n_boxes != 0.0:
            car_focal_loss /= batch_size*n_boxes
        else: 
            print(f'Division by zero encountered on cars!')
            car_focal_loss = 0.0 


        # Calculate regression loss:
        loc_loss_x = self.smooth_l1_loss(dx_tensor, torch.zeros_like(dx_tensor))
        loc_loss_y = self.smooth_l1_loss(dy_tensor, torch.zeros_like(dx_tensor))
        width_loss = self.smooth_l1_loss(dw_tensor, torch.zeros_like(dw_tensor))
        length_loss = self.smooth_l1_loss(dl_tensor, torch.zeros_like(dl_tensor))

        # Calculate classification loss:
        total_loc_loss = loc_loss_x + loc_loss_y + width_loss + length_loss

        # Calculate regression loss:
        total_loss = self.beta_loc*total_loc_loss + self.beta_cls*(background_focal_loss + car_focal_loss)

        return total_loss
    