#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt 
import cv2

# Read in image
image = cv2.imread('images/water_balloons.jpg')

# Convert to RGB
image_RGB = np.copy(image)
image_RGB = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2RGB)

# Plot RGB channels
r = image_RGB[:,:,0]
g = image_RGB[:,:,1]
b = image_RGB[:,:,2]
f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,10))
ax1.set_title("Red")
ax1.imshow(r, cmap='gray')
ax2.set_title("Blue")
ax2.imshow(r, cmap='gray')
ax3.set_title("Green")
ax3.imshow(r, cmap='gray')
plt.show()

# Convert from RGB to HSV
image_HSV = np.copy(image)
image_HSV = cv2.cvtColor(image_HSV, cv2.COLOR_BGR2HSV)

# Plot HSV channels
h = image_HSV[:,:,0]
s = image_HSV[:,:,1]
v = image_HSV[:,:,2]
f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,10))
ax1.set_title("Hue")
ax1.imshow(h, cmap='gray')
ax2.set_title("Saturation")
ax2.imshow(s, cmap='gray')
ax3.set_title("Value")
ax3.imshow(v, cmap='gray')
plt.show()

# Define pink and hue selection thresholds in RGB and HSV
lower_pink = np.array([180,0,100])
upper_pink = np.array([255,255,230])
lower_hue = np.array([160,0,0])
upper_hue = np.array([180,255,255])

# Mask the image using RGB
mask_rgb = cv2.inRange(image_RGB, lower_pink, upper_pink)
masked_image_rgb = np.copy(image_RGB)
masked_image_rgb[mask_rgb == 0] = [0,0,0]

# Mask the image using HSV
mask_hsv = cv2.inRange(image_HSV, lower_hue, upper_hue)
masked_image_hsv = np.copy(image_HSV)
masked_image_hsv[mask_hsv == 0] = [0,0,0]

plt.imshow(masked_image_hsv)

plt.show()




# Display the image
plt.imshow(image_RGB)

plt.show()