"""
A basic implementation of AlexNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from math import sqrt
import numpy as np

class OurAlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(1)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc6 = nn.Linear(in_features=9216, out_features=4096)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096)
        self.fc8 = nn.Linear(in_features=4096, out_features=1000)
        self.fc9 = nn.Linear(in_features=1000, out_features=3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.pool(y)
        y = self.conv2(y)
        y = self.pool(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.pool(y)
        y = self.flatten(y)
        y = self.fc6(y)
        y = self.fc7(y)
        y = self.fc8(y)
        y = self.fc9(y)
        y = self.softmax(y)
        return y
    