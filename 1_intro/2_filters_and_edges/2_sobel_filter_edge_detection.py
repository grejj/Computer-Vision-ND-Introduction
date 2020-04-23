#!/usr/bin/env python3

import numpy as np 
import matplotlib.pyplot as plt
import cv2

# read in the image
image = cv2.imread("images/city_hall.jpg")
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

# convert to grayscale
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
#plt.imshow(gray, cmap='gray')

# create kernel for vertical edge detection (Sobel filter)
sobel_x = np.array([[-1,0,1],
                   [-2,0,2],
                   [-1,0,1]])

# perfrom convolution of kernel and image
filtered_image = cv2.filter2D(gray, -1, sobel_x)
#plt.imshow(filtered_image, cmap='gray')

# create binary image using threshold
retval, binary_image = cv2.threshold(filtered_image, 100, 255, cv2.THRESH_BINARY)
plt.imshow(binary_image, cmap='gray')

# can be seen that high-pass filter has introduced extra noise which could be reduced by low-pass filter

plt.show()