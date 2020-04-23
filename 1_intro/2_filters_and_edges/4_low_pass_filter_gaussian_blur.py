#!/usr/bin/env python3

import numpy as np 
import matplotlib.pyplot as plt
import cv2

# read in the image
image = cv2.imread("images/brain_MR.jpg")
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

# convert to grayscale
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

# low pass filter using gaussian blur
gray_blur = cv2.GaussianBlur(gray, (5,5), 0)

# apply high pass sobel filter to compare
filtered = cv2.Sobel(gray, -1, 1, 0, ksize=3)
filtered_blur = cv2.Sobel(gray_blur, -1, 1, 0, ksize=3)
retval, filtered_binary = cv2.threshold(filtered, 50, 255, cv2.THRESH_BINARY)
retval, filtered_blur_binary = cv2.threshold(filtered_blur, 50, 255, cv2.THRESH_BINARY)


f,x = plt.subplots(3,2)
x[0,0].imshow(gray, cmap='gray'), x[0,0].set_title('input')
x[0,1].imshow(gray_blur, cmap='gray'), x[0,1].set_title('gaussian blur')
x[1,0].imshow(filtered, cmap='gray'), x[1,0].set_title('filtered')
x[1,1].imshow(filtered_blur, cmap='gray'), x[1,1].set_title('filtered blur')
x[2,0].imshow(filtered_binary, cmap='gray'), x[2,0].set_title('filtered binary')
x[2,1].imshow(filtered_blur_binary, cmap='gray'), x[2,1].set_title('filtered blur binary')
plt.show()