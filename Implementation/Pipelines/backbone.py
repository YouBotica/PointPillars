
import os
import pdb
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import pdb


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, L, stride, device):
        super(Block, self).__init__()
        self.to(device)
        layers = []
        # First layer with the specified stride
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Subsequent layers with stride 1
        for _ in range(1, L):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, stride, output_padding, device):
        super(UpSample, self).__init__()
        # Assuming stride_out is always half of stride_in based on the diagram
        self.to(device)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=output_padding),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.up(x)


class BackBone(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(BackBone, self).__init__()
        self.device = device
        self.to(self.device)
        # Define blocks with arbitrary L for now (can be tuned based on requirements)
        self.block1 = Block(in_channels, out_channels*2, L=3, stride=1, device=device)
        self.block2 = Block(out_channels*2, out_channels*2, L=3, stride=2, device=device)
        self.block3 = Block(out_channels*2, out_channels*2, L=3, stride=2, device=device)
        

        # Define upsampling layers        
        self.up1 = UpSample(out_channels*2, out_channels*2, stride=1, output_padding=0, device=device)
        self.up2 = UpSample(out_channels*2, out_channels*2, stride=2, output_padding=1, device=device)
        self.up3 = UpSample(out_channels*2, out_channels*2, stride=4, output_padding=3, device=device)
        
    def forward(self, x):
        x.to(self.device)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x2.size()
        # Upsample and concatenate
        up_x1 = self.up1(x1)   
        up_x2 = self.up2(x2)
        up_x3 = self.up3(x3)     
        concat_features = torch.cat([up_x1, up_x2, up_x3], dim=1)
        
        return concat_features
