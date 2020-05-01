#!/usr/bin/env python3

# Define and train CNN to classify FashionMNIST database

import cv2
import matplotlib.pyplot as plt
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

# 1. Load Training Data

# The output of torchvision datasets are PILImage images of range [0, 1]. 
# We transform them to Tensors for input into a CNNimport torchvision
data_transform = transforms.ToTensor()
train_data = FashionMNIST(root='./data', train=True,
                                   download=True, transform=data_transform)
print('Train data, number of images: ', len(train_data))

# 2. Data Iteration and Batching

batch_size = 20
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# image classes
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 3. Visualize training data in a batch

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(batch_size):
    ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(classes[labels[idx]])
#plt.show()

# 4. View pixel values of a normalized training image

# select an image by index
idx = 2
img = np.squeeze(images[idx])

# display the pixel values in that image
fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')
#plt.show()

# 5. Define Network Architecture

