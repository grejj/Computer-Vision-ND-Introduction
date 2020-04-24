#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import cv2

# read in the image
image = cv2.imread("images/multi_faces.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# load in cascade classifier - fully trained face detect architecture
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# run the detector on the grayscale image (image, scaleFactor, minNeighbors)
# detect more faces with smaller scale factor and lower min neighbors, but not as good matches
faces = face_cascade.detectMultiScale(gray, 4, 6)

# print out the detections found
print ('We found ' + str(len(faces)) + ' faces in this image')

# draw bounding boxes on image around faces
img_with_detections = np.copy(image)   
for (x,y,w,h) in faces:
    cv2.rectangle(img_with_detections,(x,y),(x+w,y+h),(255,0,0),5)  

f,x = plt.subplots(2,2)
x[0,0].imshow(image), x[0,0].set_title('image')
x[0,1].imshow(gray, cmap='gray'), x[0,1].set_title('gray')
x[1,0].imshow(img_with_detections), x[1,0].set_title('detections')
plt.show()