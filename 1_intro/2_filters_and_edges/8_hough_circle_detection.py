#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import cv2

# read in the image
image = cv2.imread("images/round_farms.jpg")
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

# gaussian blur
gray_blur = cv2.GaussianBlur(gray, (3,3), 0)

# find circles using Hough Circles Function
circles_im = np.copy(image)
dp = 1              # the inverse ratio of resolution
minDist = 50        # min distance between detected centers
param1 = 85         # upper threshold for internal canny edge detector
param2 = 12         # threshold for center detection, smaller value = more circles detected
min_radius = 20     # min radius to be detected, if unknown put 0 as default
max_radius = 30     # max radius to be detected, if unknown put 0 as default

# outputs circles as [x,y,r]
circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
circles = np.uint16(np.around(circles))

# draw detected circles on image
for i in circles[0,:]:
    # draw outer circle
    cv2.circle(circles_im, (i[0],i[1]), i[2], (0,255,0), 2)
    # draw the center of the circle
    cv2.circle(circles_im, (i[0],i[1]), 2, (0,0,255), 3)


f,x = plt.subplots(2,2)
x[0,0].imshow(image_copy, cmap='gray'), x[0,0].set_title('image_copy')
x[0,1].imshow(gray, cmap='gray'), x[0,1].set_title('gray')
x[1,0].imshow(gray_blur, cmap='gray'), x[1,0].set_title('gray_blur')
x[1,1].imshow(circles_im, cmap='gray'), x[1,1].set_title('circles_im')
plt.show()