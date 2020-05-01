#!/usr/bin/env python3

# Visualize four filtered feature maps of a convolution layer

import cv2
import matplotlib.pyplot as plt
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. import the image

# load image
image = cv2.imread('images/udacity_sdc.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# normalize image
gray = gray.astype("float32")/255
# plot image
#plt.imshow(gray, cmap='gray')
#plt.show()

# 2. setup filters

filters = np.array([[-1,-1,1,1],[-1,-1,1,1],[-1,-1,1,1],[-1,-1,1,1]])
print('Filter shape: ', filters.shape)
# define four filters as linear combinations of above
filter_1 = filters
filter_2 = -filter_1
filter_3 = filter_1.T 
filter_4 = -filter_3
filters = np.array([filter_1,filter_2,filter_3,filter_4])
# visualize four filters
fig = plt.figure(figsize=(10, 5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))
    width, height = filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(filters[i][x][y]), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if filters[i][x][y]<0 else 'black')
#plt.show()

# 3. define convolutional layers
class Network(nn.Module):
    def __init__(self, weight):
        super(Network, self).__init__()
        # initialize weights of convolutional layer to be filters defined above
        height, width = weight.shape[2:]
        # assumes 4 grayscale filters
        self.conv = nn.Conv2d(1,4,kernel_size=(height,width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        # calculates output of convolutional layer
        # pre and post activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)

        return conv_x, activated_x

# setup model and weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Network(weight)
#print(model)

# 4. Visualize output of each filter

# helper function for visualizing output of given layer
def viz_layer(layer, n_filters=4):
    figure = plt.figure(figsize=(20,20))
    for i in range(n_filters):
        ax = figure.add_subplot(1, n_filters, i+1, xticks=[], yticks=[])
        # grab layer outputs
        ax.imshow(np.squeeze(layer[0,i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i+1))

# plot original image
plt.imshow(gray, cmap='gray')

# visualize all filters
fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(left=0, right=1.5, bottom=0.8, top=1, hspace=0.05, wspace=0.05)
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))
    
# convert the image into an input Tensor
gray_img_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(1)

# get the convolutional layer (pre and post activation)
conv_layer, activated_layer = model(gray_img_tensor)

# visualize the output of a conv layer
viz_layer(conv_layer)
viz_layer(activated_layer)
plt.show()