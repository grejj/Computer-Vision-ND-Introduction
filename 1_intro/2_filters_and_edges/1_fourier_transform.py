#!/usr/bin/env python3

# The frequency components of an image can be displayed after doing a Fourier Transform (FT). An FT looks at the components of an image (edges that are high-frequency, and areas of smooth color as low-frequency), and plots the frequencies that occur as points in spectrum.

import numpy as np 
import matplotlib.pyplot as plt
import cv2

# read in the images
image_stripes = cv2.imread("images/stripes.jpg")
image_stripes = cv2.cvtColor(image_stripes, cv2.COLOR_BGR2RGB)

image_solid = cv2.imread("images/pink_solid.jpg")
image_solid = cv2.cvtColor(image_solid, cv2.COLOR_BGR2RGB)

# display images
#f, (ax1,ax2) = plt.subplots(1, 2, figsize=(10,5))
#ax1.imshow(image_stripes)
#ax2.imshow(image_solid)
#plt.show()

# convert to grayscale to focus on intensity patterns
gray_stripes = cv2.cvtColor(image_stripes, cv2.COLOR_RGB2GRAY)
gray_solid = cv2.cvtColor(image_solid, cv2.COLOR_RGB2GRAY)

# normalize image color values between 0 and 1
norm_stripes = gray_stripes / 255.0
norm_solid = gray_solid / 255.0

# perfrom fast fourier transform and create a scaled, frequency transfrom image
def ft_image(norm_image):
    ''' This function takes in a normalized, grayscale image and returns a frequency spectrum transform of that image.'''
    f = np.fft.fft2(norm_image)
    fshift = np.fft.fftshift(f)
    frequency_tx = 20*np.log(np.abs(fshift))
    return frequency_tx

# get fourier transforms of images
f_stripes = ft_image(norm_stripes)
f_solid = ft_image(norm_solid)

# display the images
# original images to the left of their frequency transform
# f, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(20,10))
# ax1.set_title('original image')
# ax1.imshow(image_stripes)
# ax2.set_title('frequency transform image')
# ax2.imshow(f_stripes, cmap='gray')
# ax3.set_title('original image')
# ax3.imshow(image_solid)
# ax4.set_title('frequency transform image')
# ax4.imshow(f_solid, cmap='gray')
# plt.show()

# try this on real-world image
image = cv2.imread('images/birds.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
norm_image = gray/255.0
f_image = ft_image(norm_image)

# display image
f, (ax1,ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(image)
ax2.imshow(f_image, cmap='gray')
plt.show()