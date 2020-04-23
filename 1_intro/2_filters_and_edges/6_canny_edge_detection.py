#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import cv2

# read in the image
image = cv2.imread("images/sunflower.jpg")
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

# different canny implementations based on hysteresis threshold values 
edges = cv2.Canny(gray, 120, 240)
wide_edges = cv2.Canny(gray, 30, 100)
tight_edges = cv2.Canny(gray, 180, 240)

f,x = plt.subplots(2,2)
x[0,0].imshow(image_copy, cmap='gray'), x[0,0].set_title('image_copy')
x[0,1].imshow(edges, cmap='gray'), x[0,1].set_title('edges')
x[1,0].imshow(wide_edges, cmap='gray'), x[1,0].set_title('wide_edges')
x[1,1].imshow(tight_edges, cmap='gray'), x[1,1].set_title('tight_edges')
plt.show()