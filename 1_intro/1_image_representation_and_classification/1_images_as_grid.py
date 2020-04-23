#!/usr/bin/env python3

import numpy as np 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import cv2

# Read in image
image = mpimg.imread('images/waymo_car.jpg')

# Print image dimensions
print('Image dimensions: {}'.format(image.shape))

# Change color to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray_image, cmap='gray')

# Print specific gray scale value
x = 400
y = 300
print(gray_image[y,x])

# Finds min and max grayscale values in image
max_val = np.amax(gray_image)
min_val = np.amin(gray_image)
print('Max: {} | Min: {}'.format(max_val, min_val))

# create a 5x5 image using just grayscale, numerical values
tiny_image = np.array([[0, 20, 30, 150, 120],
                      [200, 200, 250, 70, 3],
                      [50, 180, 85, 40, 90],
                      [240, 100, 50, 255, 10],
                      [30, 0, 75, 190, 220]])

# show pixel grid
plt.matshow(tiny_image, cmap='gray')
plt.show()