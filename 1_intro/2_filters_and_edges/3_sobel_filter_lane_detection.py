#!/usr/bin/env python3

import numpy as np 
import matplotlib.pyplot as plt
import cv2

def dir_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    # 1) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # 2) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    # 3) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    gradient_dir = np.arctan2(abs_sobely, abs_sobelx)

    print("Gradient dir: ", gradient_dir)
        
    # 4) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(gradient_dir)
    dir_binary[(gradient_dir >= thresh[0]) & (gradient_dir <= thresh[1])] = 1
    
    # 5) Return this mask as your binary_output image    
    return dir_binary


# read in the image
image = cv2.imread("images/curved_lane.jpg")
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

# convert to grayscale
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)


# 3x3 sobel x filter
sobel_3x = np.array([[-1,0,1],
                   [-2,0,2],
                   [-1,0,1]])
sobel_3x_image = cv2.filter2D(gray, -1, sobel_3x)

# 3x3 sobel y filter
sobel_3y = np.array([[-1,-2,-1],
                   [0,0,0],
                   [1,2,1]])
sobel_3y_image = cv2.filter2D(gray, -1, sobel_3y)

# 3x3 sobel x and y filter
sobel_3xy_image = cv2.filter2D(sobel_3y_image, -1, sobel_3x)

# 5x5 sobel x filter
sobel_5x = np.array([[-2,-1,0,1,2],
                   [-2,-1,0,1,2],
                   [-4,-2,0,2,4],
                   [-2,-1,0,1,2],
                   [-2,-1,0,1,2]])
sobel_5x_image = cv2.filter2D(gray, -1, sobel_5x)

# 5x5 sobel y filter
sobel_5y = np.array([[-2,-2,-4,-2,-2],
                   [-1,-1,-2,-1,-1],
                   [0,0,0,0,0],
                   [1,1,2,1,1],
                   [2,2,4,2,2]])
sobel_5y_image = cv2.filter2D(gray, -1, sobel_5y)

# 5x5 sobels 
sobel_0 = cv2.Sobel(gray, -1, 0, 1, ksize=5)
sobel_45 = cv2.Sobel(gray, -1, 1, 1, ksize=5)
sobel_63 = cv2.Sobel(gray, -1, 2, 1, ksize=5)
sobel_90 = cv2.Sobel(gray, -1, 1, 0, ksize=5)

f,x = plt.subplots(2,4)
x[0,0].imshow(gray, cmap='gray'), x[0,0].set_title('input')
x[0,1].imshow(sobel_3x_image, cmap='gray'), x[0,1].set_title('sobel_3x_image')
x[0,2].imshow(sobel_3y_image, cmap='gray'), x[0,2].set_title('sobel_3y_image')
x[0,3].imshow(sobel_3xy_image, cmap='gray'), x[0,3].set_title('sobel_3xy_image')
x[1,0].imshow(sobel_0, cmap='gray'), x[1,0].set_title('sobel_0')
x[1,1].imshow(sobel_45, cmap='gray'), x[1,1].set_title('sobel_45')
x[1,2].imshow(sobel_63, cmap='gray'), x[1,2].set_title('sobel_63')
x[1,3].imshow(sobel_90, cmap='gray'), x[1,3].set_title('sobel_90')
plt.show()