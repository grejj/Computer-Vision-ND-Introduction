#!/usr/bin/env python3

import matplotlib.pyplot as plt 
import numpy as np 
import cv2 

def orientations(contours):
    """
    Orientation 
    :param contours: a list of contours
    :return: angles, the orientations of the contours
    """
    angles = []
    for contour in contours:
        # fit an ellipse to a contour and extract the angle from ellipse
        (x,y), (MA,ma), angle = cv2.fitEllipse(contour)
        angles.append(angle)
    
    return angles

def left_hand_crop(image, contour):
    """
    Left hand crop 
    :param image: the original image
    :param contour: the contour that will be used for cropping
    :return: cropped_image, the cropped image around the left hand
    """
    cropped_image = np.copy(image)
    x,y,w,h = cv2.boundingRect(contour)
    cropped_image = cropped_image[y:y+h,x:x+w]
    return cropped_image

# read in the image
image = cv2.imread("images/thumbs_up_down.jpg")
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

# create a binary thresholded image
retval, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

# find contour of thresholded image
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# draw contours on image
contour_image = np.copy(image_copy)
contour_image = cv2.drawContours(contour_image, contours, -1, (0,255,0), 2)

# get orientation of hand
angles = orientations(contours)
print('Angles of each contour: {}'.format(str(angles)))

# get cropped image around left hand
cropped_image = left_hand_crop(image_copy, contours[1])

f,x = plt.subplots(2,2)
x[0,0].imshow(image_copy), x[0,0].set_title('image')
x[0,1].imshow(binary, cmap='gray'), x[0,1].set_title('binary')
x[1,0].imshow(contour_image, cmap='gray'), x[1,0].set_title('contour_image')
x[1,1].imshow(cropped_image), x[1,1].set_title('left hand')

plt.show()