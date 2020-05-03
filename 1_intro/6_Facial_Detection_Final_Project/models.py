## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        
        # conv layer with input 1 channel image grayscale and 32 output feature maps
        # 5x5 convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)    # (32,220,220)
        # pooling layer with kernel size 2x2 and stride of 2
        self.pool1 = nn.MaxPool2d(2, 2)     # (32,110,110)
        
        # conv layer with 32 input feature maps and 64 output feature maps
        # 3x3 convolution kernel
        self.conv2 = nn.Conv2d(32, 64, 3)   # (64,108,108)
        # pooling layer with kernel size 2x2 and stride of 2
        self.pool2 = nn.MaxPool2d(2, 2)     # (64,54,54)

        # conv layer with 64 input feature maps and 128 output feature maps
        # 3x3 convolution kernel
        self.conv3 = nn.Conv2d(64, 128, 3)   # (128,52,52)
        # pooling layer with kernel size 2x2 and stride of 2
        self.pool3 = nn.MaxPool2d(2, 2)     # (128,26,26)

        # conv layer with 128 input feature maps and 256 output feature maps
        # 3x3 convolution kernel
        self.conv4 = nn.Conv2d(128, 256, 3)   # (256,24,24)
        # pooling layer with kernel size 2x2 and stride of 2
        self.pool4 = nn.MaxPool2d(2, 2)     # (256,12,12)

        # Linear layer output 6000 values
        self.fc1 = nn.Linear(256*12*12, 6000) # (6000)
        # Linear layer output 1000 values
        self.fc2 = nn.Linear(6000, 1000) # (1000)
        # Linear layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        self.fc3 = nn.Linear(1000, 136) # (136)

        # dropout layers to reduce overfitting
        # probability increases with each layer on 0.1 increments
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        self.drop6 = nn.Dropout(p = 0.6)
        
    def forward(self, x):
        # Define forward pass
        
        # convlutuion, activation, and pooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.drop4(x)

        # flatten 
        x = x.view(x.size(0), -1)

        # dense layers
        x = self.fc1(x)
        x = self.drop5(x)
        x = self.fc2(x)
        x = self.drop6(x)
        x = self.fc3(x)

        return x
