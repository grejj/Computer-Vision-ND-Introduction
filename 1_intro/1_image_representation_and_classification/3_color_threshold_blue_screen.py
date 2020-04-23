#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt 
import cv2

# Read in image
image_BGR = cv2.imread('images/pizza_bluescreen.jpg')

# Print out image data
print('Image is: {} with dimensions: {}'.format(type(image_BGR), image_BGR.shape))

# Make a copy of image as RGB
image_RGB = np.copy(image_BGR)
image_RGB = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2RGB)

# Define color threshold for blue
lower_blue = np.array([0,0,220])
upper_blue = np.array([50,70,255])

# Define the masked area
mask = cv2.inRange(image_RGB, lower_blue, upper_blue)

# Visualize the mask
plt.imshow(mask, cmap='gray')

# Create masked image
masked_image = np.copy(image_RGB) 
masked_image[mask != 0] = [0,0,0]

# Mask and add background image
background_image = cv2.imread('images/space_background.jpg')
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

# Crop background to same size as original image
crop_background = background_image[0:514, 0:816]

# Mask the cropped background so no pizza area is blocked
crop_background[mask == 0] = [0,0,0]

complete_image = crop_background + masked_image

# Display the image
#plt.imshow(image_RGB)
#plt.imshow(image_BGR)
#plt.imshow(mask, cmap='gray')
#plt.imshow(masked_image)
plt.imshow(complete_image)

plt.show()