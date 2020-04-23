#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import cv2

# read in the image
image = cv2.imread("images/phone.jpg")
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

# perform canny edge detection
edges = cv2.Canny(gray, 50, 100)

# find lines using Hough transform
rho = 1             # 1 pixel resolution
theta = np.pi/180   # 1 degree resolution
threshold = 60      # min number of Hough space intersections for a line
min_line_length = 50
min_line_gap = 5

# outputs lines as [x1,y1,x2,y2]
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, min_line_gap)

# draw detected lines on image
line_image = np.copy(image_copy)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

f,x = plt.subplots(2,2)
x[0,0].imshow(image_copy, cmap='gray'), x[0,0].set_title('image_copy')
x[0,1].imshow(gray, cmap='gray'), x[0,1].set_title('gray')
x[1,0].imshow(edges, cmap='gray'), x[1,0].set_title('edges')
x[1,1].imshow(line_image, cmap='gray'), x[1,1].set_title('line_image')
plt.show()