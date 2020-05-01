#!/usr/bin/env python3

# Load a trained CNN from FashionMNIST and implement feature visualization to see what features network learned to extract

import cv2
import matplotlib.pyplot as plt
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

# 1. Load Training Data

# The output of torchvision datasets are PILImage images of range [0, 1]. 
# We transform them to Tensors for input into a CNNimport torchvision
data_transform = transforms.ToTensor()
test_data = FashionMNIST(root='./data', train=False,
                                  download=True, transform=data_transform)
print('Test data, number of images: ', len(test_data))

# 2. Data Iteration and Batching

batch_size = 20
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
# image classes
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 3. Visualize training data in a batch

# obtain one batch of training images
#dataiter = iter(test_loader)
#images, labels = dataiter.next()
#images = images.numpy()

# plot the images in the batch, along with the corresponding labels
#fig = plt.figure(figsize=(25, 4))
#for idx in np.arange(batch_size):
#    ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])
#    ax.imshow(np.squeeze(images[idx]), cmap='gray')
#    ax.set_title(classes[labels[idx]])
#plt.show()


# 4. Define Network Architecture
class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()

        # 1 input image channel, 10 output feature maps
        # 3x3 kernel
        # so output tensor will be (10, 26, 26)
        self.conv1 = nn.Conv2d(1, 10, 3)

        # maxpool layer with kernel=2 and stride=2
        # output tensor will be (10, 13, 13)
        self.pool = nn.MaxPool2d(2, 2)

        # second conv layer has 10 inputs, 20 outputs, 3x3 kernel
        # output tensor will be (20, 11, 11)
        self.conv2 = nn.Conv2d(10, 20, 3)

        # another maxpool layer converts tensor to (20, 5, 5)

        # final linear layer has 20*(5*5) inputs and 50 outputs
        self.fc1 = nn.Linear(20*5*5, 50)

        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # finally, create 10 output channels (for the 10 classes)
        self.fc2 = nn.Linear(50, 10)


    # feedforward behavior
    def forward(self, x):
        # two conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # flatten 
        x = x.view(x.size(0), -1)

        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)

        return x
    
network = Network()

# 4. Load the model
model_dir = 'saved_models/'
model_name = 'fashion_net_ex.pt'
network.load_state_dict(torch.load(model_dir+model_name))
print(network)

# 5. Visualize filters (weights) in first conv layer

# Get the weights in the first conv layer
weights = network.conv1.weight.data
w = weights.numpy()
# for 10 filters
fig=plt.figure(figsize=(20, 8))
columns = 5
rows = 2
for i in range(0, columns*rows):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(w[i][0], cmap='gray')
print('First convolutional layer')
plt.show()

# 5. Apply filters to test image to produce activation maps for first and second layers

# obtain one batch of testing images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images = images.numpy()
# select an image by index
idx = 3
img = np.squeeze(images[idx])

# first conv layer for 10 filters
fig=plt.figure(figsize=(30, 10))
print(w.shape)
columns = 5*2
rows = 2
for i in range(0, columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if ((i%2)==0):
        plt.imshow(w[int(i/2)][0], cmap='gray')
    else:
        c = cv2.filter2D(img, -1, w[int((i-1)/2)][0])
        plt.imshow(c, cmap='gray')
print('First convolutional layer activation maps')
plt.show()

# second conv layer for 20 filters
fig=plt.figure(figsize=(30, 10))
weights = network.conv2.weight.data
w = weights.numpy()
print(w.shape)
columns = 5*2
rows = 2*2
for i in range(0, columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if ((i%2)==0):
        plt.imshow(w[int(i/2)][0], cmap='gray')
    else:
        c = cv2.filter2D(img, -1, w[int((i-1)/2)][0])
        plt.imshow(c, cmap='gray')
print('Second convolutional layer activation maps')
plt.show()