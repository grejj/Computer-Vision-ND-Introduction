#!/usr/bin/env python3

# Define and train CNN to classify FashionMNIST database using dropout layers and momentum

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
train_data = FashionMNIST(root='./data', train=True,
                                   download=True, transform=data_transform)
test_data = FashionMNIST(root='./data', train=False,
                                  download=True, transform=data_transform)
print('Train data, number of images: ', len(train_data))
print('Test data, number of images: ', len(test_data))

# 2. Data Iteration and Batching

batch_size = 20
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
# image classes
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 3. Visualize training data in a batch

# obtain one batch of training images
#dataiter = iter(train_loader)
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
print(network)

# 5. Specify Optimizer and Loss Function

# cross entropy loss = NLL loss + softmax
criterion = nn.CrossEntropyLoss()
# using SGD with small learning rate and some momentum
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

# 6. Get Accuracy of Network Before Training

correct = 0
total = 0
for images, labels in test_loader:
    # forward pass
    outputs = network(images)
    # get predicted class
    _, predicted = torch.max(outputs.data, 1)
    # update correct and total amounts
    total += labels.size(0)
    correct += (predicted == labels).sum()
# get accuracy
accuracy = 100.0 * correct.item() / total
print('Accuracy before training: {}'.format(accuracy))

# 7. Train the Network

def train(n_epochs):
    loss_over_time = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        for batch, data in enumerate(train_loader):
            # get input images and labels
            inputs, labels = data
            # zero parameter gradients
            optimizer.zero_grad()
            # forward pass
            outputs = network(inputs)
            # calculate loss
            loss = criterion(outputs, labels)
            # backward pass, calculate parameter gradients
            loss.backward()
            # update parameters
            optimizer.step()
            # update loss
            running_loss += loss.item()
            # print every 1000 batches
            if batch % 1000 == 999:
                average_loss = running_loss/1000
                loss_over_time.append(average_loss)
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch+1, average_loss))
                running_loss = 0.0

    return loss_over_time

training_loss = train(30)
# visualize the loss as the network trained
#plt.plot(training_loss)
#plt.xlabel('1000\'s of batches')
#plt.ylabel('loss')
#plt.ylim(0, 2.5) # consistent scale
#plt.show()

# 8. Get Accuarcy of Training Network

# initialize tensor and lists to monitor test loss and accuracy
test_loss = torch.zeros(1)
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

# set the module to evaluation mode
network.eval()

for batch_i, data in enumerate(test_loader):
    
    # get the input images and their corresponding labels
    inputs, labels = data
    
    # forward pass to get outputs
    outputs = network(inputs)

    # calculate the loss
    loss = criterion(outputs, labels)
            
    # update average test loss 
    test_loss = test_loss + ((torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))
    
    # get the predicted class from the maximum value in the output-list of class scores
    _, predicted = torch.max(outputs.data, 1)
    
    # compare predictions to true label
    # this creates a `correct` Tensor that holds the number of correctly classified images in a batch
    correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))
    
    # calculate test accuracy for *each* object class
    # we get the scalar value of correct items for a class, by calling `correct[i].item()`
    for i in range(batch_size):
        label = labels.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

print('Test Loss: {:.6f}\n'.format(test_loss.numpy()[0]))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

        
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

# 9. Visualize Results

# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
# get predictions
preds = np.squeeze(network(images).data.max(1, keepdim=True)[1].numpy())
images = images.numpy()

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(batch_size):
    ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx] else "red"))
#plt.show()

# 10. Save the model
model_dir = 'saved_models/'
model_name = 'fashion_net_ex.pt'
torch.save(network.state_dict(), model_dir+model_name)


# Final Note
''' 
Adding dropout layers and momentum has definitely improved performance. Shirts, pullovers, and coats all
have similar shape so they have the least accuracy. This could be improved even further by perhaps doing
some data augmentation or adding another conv layer to extract higher level features.
'''