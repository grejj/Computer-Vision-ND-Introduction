#!/usr/bin/env python3

import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

# read in image
image = cv2.imread("images/rainbow_flag.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

level_1 = cv2.pyrDown(image)
level_2 = cv2.pyrDown(level_1)
level_3 = cv2.pyrDown(level_2)

f,x = plt.subplots(2,2)
x[0,0].imshow(image), x[0,0].set_title('level_0')
x[0,1].imshow(level_1), x[0,1].set_title('level_1')
x[1,0].imshow(level_2), x[1,0].set_title('level_2')
x[1,1].imshow(level_3), x[1,1].set_title('level_3')
plt.show()