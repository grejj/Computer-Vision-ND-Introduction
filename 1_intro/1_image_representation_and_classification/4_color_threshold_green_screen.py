#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import cv2

# Read in image
image = mpimg.imread('images/car_green_screen.jpg')

# Print out image data
print('Image is: {} with dimensions: {}'.format(type(image), image.shape))

# Define color threshold for green
lower_green = np.array([0,100,0])
upper_green = np.array([150,255,100])

# Define the masked area
mask = cv2.inRange(image, lower_green, upper_green)

# Visualize the mask
plt.imshow(mask, cmap='gray')

# Create masked image
masked_image = np.copy(image) 
masked_image[mask != 0] = [0,0,0]

# Visualize masked image
plt.imshow(masked_image, cmap='gray')

# Mask and add background image
background_image = cv2.imread('images/sky.jpg')
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

# Crop background to same size as original image
crop_background = background_image[0:450, 0:660]

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