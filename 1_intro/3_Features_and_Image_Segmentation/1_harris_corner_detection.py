#!/usr/bin/env python3

import matplotlib.pyplot as plt 
import numpy as np 
import cv2 

# read in the image
image = cv2.imread("images/waffle.jpg")
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

# convert to float type for harris detector
gray = np.float32(gray)

# detect corners
window_size = 2
sobel_kernel_size = 3
k = 0.04
corner_detect = cv2.cornerHarris(gray, 2, 3, k)

# dilate corner image to enhance corner points
dilate_corner_detect = cv2.dilate(corner_detect, None)

# select and display strongest corners
threshold = 0.05 * dilate_corner_detect.max()
final_corner_detect = np.copy(image_copy)

for j in range(0, dilate_corner_detect.shape[0]):
    for i in range(0, dilate_corner_detect.shape[1]):
        if (dilate_corner_detect[j,i] > threshold):
            cv2.circle(final_corner_detect, (i,j), 2, (0,255,0), 1)

f,x = plt.subplots(2,2)
x[0,0].imshow(image_copy), x[0,0].set_title('image')
x[0,1].imshow(corner_detect, cmap='gray'), x[0,1].set_title('corner detect')
x[1,0].imshow(dilate_corner_detect, cmap='gray'), x[1,0].set_title('dilated corner detect')
x[1,1].imshow(final_corner_detect, cmap='gray'), x[1,1].set_title('final corner detect')

plt.show()