# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:13:55 2024

@author: Mateo-drr
"""

import torch.nn as nn
import torch.nn.functional as F
from src.blocks.SEB import SEBlock


class MER(nn.Module):
    def __init__(self, in_channels=1, n_class: int = 56, dropout: float = 0.2) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.InstanceNorm2d(64, affine=True),
            nn.Mish(),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.conv3 = nn.Sequential(
            nn.InstanceNorm2d(64, affine=True),
            nn.Mish(),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.Mish(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.filter1 = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256, affine=True),
            nn.Mish(),
            nn.Dropout2d(dropout),
        )
        self.filter2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
        )
        self.filter3 = nn.Sequential(
            nn.InstanceNorm2d(256, affine=True),
            nn.Mish(),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, 256, 3, 1, 1),
        )
        self.filter4 = nn.Sequential(
            nn.InstanceNorm2d(256, affine=True),
            nn.Mish(),
            nn.Dropout2d(dropout),
            SEBlock(256, reduce_dim=16),
        )

        ksize = 3
        pad = (ksize - 1) // 2
        self.lnormf = nn.LayerNorm(256)
        self.lnormt = nn.LayerNorm(256)
        self.attfreq = nn.Sequential(
            nn.Conv2d(256, 256, (ksize, 1), stride=1, padding=(pad, 0)),
            nn.Dropout2d(dropout),
        )

        self.atttime = nn.Sequential(
            nn.Conv2d(256, 256, (1, ksize), stride=1, padding=(0, pad)),
            nn.Dropout2d(dropout),
        )

        self.fco = nn.Sequential(
            nn.LayerNorm([256, 1]),
            nn.Mish(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(256, 1024, 1, 1, 0),
            nn.BatchNorm1d(1024),
            nn.Mish(inplace=True),
        )
        self.output = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(1024, n_class, 1, 1, 0),
        )

        self.init_weights()

    def init_weights(self):
        """Initialize model weights using Kaiming initialization for ReLU."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d)):  # Apply to Conv2d and Linear layers
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x) + x

        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.filter1(x)
        x = self.filter2(x)
        x = self.filter3(x) + x
        x = self.filter4(x)  # b, 256, 8,32 -> b, 256, 4 ,16

        # input is [b,c,freq,time]

        # Each time step (16) has a weight vector for the (4) frequencies
        weights = F.softmax(
            self.attfreq(self.lnormf(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)), dim=2
        )
        # Each weighted freq column is summed (w. avg)
        x = (x * weights).sum(dim=2, keepdim=True)

        weights = F.softmax(
            self.atttime(self.lnormt(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)), dim=-1
        )
        x = (x * weights).sum(dim=-1, keepdim=True)
        # output should be [b,c,1,1]:

        x = self.fco(x.squeeze(-1))
        x = self.output(x)
        return x.squeeze(-1)

