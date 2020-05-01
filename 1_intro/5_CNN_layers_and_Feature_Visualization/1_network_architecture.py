#!/usr/bin/env python3

# Defines a simple neural network architecture with the following layers:
'''
1. Convolutional Layers
2. Maxpooling Layers
3. Fully-Connected (linear) Layers
'''

import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self, n_classes):
        super(Network, self).__init__()

        # 1 input image channel, 32 output channels/feature maps
        # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1,32,5)

        # maxpool layer
        # pool with kernel_size = 2, stride = 2
        self.pool = nn.MaxPool2d(2,2)

        # fully-connected layer 
        # 32*4 input size to account for the downsampled image size after pooling
        # num_classes = number of classes of image data
        self.fc1 = nn.Linear(32*4, n_classes)

    # define feedforward behavior
    def forward(self, x):
        # one conv/relu + pool
        x = self.pool(F.relu(self.conv1(x)))

        # prep for linear layer by flattening feature maps in feature vectors
        x = x.view(x.size(0), -1)
        # linear layer
        x = F.relu(self.fc1(x))

        return x

# print network
n_classes = 20
net = Network(n_classes)
print(net)