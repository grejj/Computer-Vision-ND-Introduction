#!/usr/bin/env python3

# Implementation of YOLOv3 algorithm for object detection
# Using a pretrained version of Darknet that has been pretrained on COCO database

import cv2
import matplotlib.pyplot as plt
from darknet import Darknet
from utils import *

# set location of config file that contains network architecture
config_file = './cfg/yolov3.cfg'
# set location of weights file that contains pre-trained network weights
weight_file = './weights/yolov3.weights'
# set location and name of COCO object classes file
name_file = './data/coco.names'

# load the network architecture
model = Darknet(config_file)
# load the pre-trained weights
model.load_weights(weight_file)
# load the COCO object classes
class_names = load_class_names(name_file)

# print network for viewing
model.print_network()

# set default figure size
plt.rcParams['figure.figsize'] = [24.0, 24.0]
# load image
image = cv2.imread('./images/surf.jpg')
# convert image to RGB
original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# resize image to input size of first layer (416x416x3)
resized_image = cv2.resize(original_image, (model.width, model.height))

# display images
plt.subplot(121)
plt.title('Original Image')
plt.imshow(original_image)
plt.subplot(122)
plt.title('Resized Image')
plt.imshow(resized_image)
plt.show()

# set non-maximal suppression threshold
# all boxes with detection probability below threshold are removed
nms_thresh = 0.6

# set intersection over union threshold
# all boxes with IOU over threshold with respect to best bouding box are removed
iou_thresh = 0.4

# input model, image, thresholds and return bouding boxes [x,y,width,height,confidence,class probability,class ID]
boxes = detect_objects(model, resized_image, iou_thresh, nms_thresh)
# print class objects found and corresponding class probability
print_objects(boxes, class_names)
# plot bounding boxes and class labels
plot_boxes(original_image, boxes, class_names, plot_labels=True) 
