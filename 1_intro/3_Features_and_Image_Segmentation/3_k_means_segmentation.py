#!/usr/bin/env python3

import matplotlib.pyplot as plt 
import numpy as np 
import cv2 

# read in the image
image = cv2.imread("images/monarch.jpg")
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

# reshape into Mx3 array where M is number of pixels in image
pixel_vals = image_copy.reshape((-1,3))
pixel_vals = np.float32(pixel_vals)

# define k-means stopping criteria = tuple of (type, max_iter, epsilon)
# EPS + MAX_ITER = stop when accuracy (epsilon) is reached or reached max iterations
max_iter = 10
eps = 1.0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)

# implement k-means clustering
k = 2
attempts = 10 # number of times executed with different initial labellings -> returning one with best compactness
compactness, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

# convert data back to 8-bit image for display
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape((image_copy.shape))
labels_reshape = labels.reshape(image_copy.shape[0], image_copy.shape[1])
 
# mask image segment
masked_image_1 = np.copy(image_copy)
masked_image_1[labels_reshape == 0] = [0,0,0]
masked_image_0 = np.copy(image_copy)
masked_image_0[labels_reshape == 1] = [0,0,0]

f,x = plt.subplots(3,2)
x[0,0].imshow(image_copy), x[0,0].set_title('image')
x[0,1].imshow(segmented_image), x[0,1].set_title('segmented')
x[1,0].imshow(labels_reshape==0, cmap='gray'), x[1,0].set_title('cluster 0')
x[1,1].imshow(labels_reshape==1, cmap='gray'), x[1,1].set_title('cluster 1')
x[2,0].imshow(masked_image_0), x[2,0].set_title('masked 0')
x[2,1].imshow(masked_image_1), x[2,1].set_title('masked 1')

plt.show()