#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import cv2
import glob
import os
import random

# This function loads in images and their labels and places them in a list
def load_dataset(image_dir):
    # Populate this empty image list
    image_list = []
    image_types = ["day", "night"]
    
    # Iterate through each color folder
    for image_type in image_types:
        
        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, image_type, "*")):
            
            # Read in the image
            image = mpimg.imread(file)
            
            # Check if the image exists/if it's been correctly read-in
            if not image is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                image_list.append((image, image_type))

    return image_list


# Return binary label for day=1, night=0
def encode(label):
    numerical_val = 0
    if(label == 'day'):
        numerical_val = 1
    # else it is night and can stay 0
    return numerical_val

# Standarizes images with correct size and label
def standardize(image_list):
    standard_list = []
    # iterate through all image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # standardize the image
        standardized_image = cv2.resize(image, (1100, 600))

        # create a numerical label
        binary_label = encode(label)

        # add standardized image and numerical label to new list
        standard_list.append((standardized_image, binary_label))

    return standard_list

def display_image_label(pair):
    image, label = pair
    plt.imshow(image)
    print("Shape: "+str(image.shape))
    print("Label [1 = day, 0 = night]: " + str(label))
    plt.show()

def plot_HSV_channels(image, figure_num):
    plt.figure(figure_num)
    image_hsv = np.copy(image)
    image_hsv = cv2.cvtColor(image_hsv, cv2.COLOR_RGB2HSV)
    h = image_hsv[:,:,0]
    s = image_hsv[:,:,1]
    v = image_hsv[:,:,2]
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(20,10))
    ax1.set_title("Image")
    ax1.imshow(image)
    ax2.set_title("Hue")
    ax2.imshow(h, cmap='gray')
    ax3.set_title("Saturation")
    ax3.imshow(s, cmap='gray')
    ax4.set_title("Value")
    ax4.imshow(v, cmap='gray')

def plot_RGB_channels(image):
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(20,10))
    ax1.set_title("Image")
    ax1.imshow(image)
    ax2.set_title("Red")
    ax2.imshow(r, cmap='gray')
    ax3.set_title("Blue")
    ax3.imshow(r, cmap='gray')
    ax4.set_title("Green")
    ax4.imshow(r, cmap='gray')
    plt.show()

# return average brightness of image
def avg_brightness(image):
    # convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # add all pixel values in v channel
    sum_brightness = np.sum(hsv[:,:,2])
    # image area
    area = hsv.shape[0] * hsv.shape[1]
    # return sum divided by area of image
    return (sum_brightness / area)

# return average brightness of image
def avg_hue(image):
    # convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # add all pixel values in h channel
    sum_hue = np.sum(hsv[:,:,0])
    # image area
    area = hsv.shape[0] * hsv.shape[1]
    # return sum divided by area of image
    return (sum_hue / area)
    
# day night classifier
def bclassifier(image, bthreshold):
    # get average image brightness
    brightness = avg_brightness(image)
    # predict label using threshold
    if (brightness > bthreshold):
        return 1
    else:
        return 0

# day night classifier
def bhclassifier(image, bthreshold, hthreshold):
    # get average image brightness
    brightness = avg_brightness(image)
    # get average image hue
    hue = avg_hue(image)
    # predict label using threshold
    if (brightness > bthreshold):
        if (hue < hthreshold):
            return 0
        else:
            return 1
    else:
        if (hue > threshold):
            return 1
        else:
            return 0

# return a list of misclassified and classified images
def get_classified_images_brightness(images, bthreshold):

    misclassified_images = []
    classified_images = []
    
    # iterate through all images and try to classify them
    for image in images:
        predicted_label = bclassifier(image[0], bthreshold)
        correct_label = image[1]

        # if labels aren't the same, add to misclassified list
        if (predicted_label != correct_label):
            misclassified_images.append((image[0], predicted_label, correct_label))
        else:
            classified_images.append((image[0], predicted_label, correct_label))

    return (classified_images, misclassified_images)

# return a list of misclassified and classified images
def get_classified_images_brightness_hue(images, bthreshold, hthreshold):

    misclassified_images = []
    classified_images = []
    
    # iterate through all images and try to classify them
    for image in images:
        predicted_label = bhclassifier(image[0], bthreshold, hthreshold)
        correct_label = image[1]

        # if labels aren't the same, add to misclassified list
        if (predicted_label != correct_label):
            misclassified_images.append((image[0], predicted_label, correct_label))
        else:
            classified_images.append((image[0], predicted_label, correct_label))

    return (classified_images, misclassified_images)
    

# image data
training_images = load_dataset("day_night_images/training/")
testing_images = load_dataset("day_night_images/test/")

# standardize image data
standardized_training_images = standardize(training_images)
standardized_testing_images = standardize(testing_images)

# display image
#display_image_label(standardized_training_images[0])

max_night_brightness = 0
min_night_brightness = 255
min_day_brightness = 255
max_day_brightness = 0
max_night_hue = 0
min_night_hue = 255
min_day_hue = 255
max_day_hue = 0

for pair in standardized_testing_images:
    image = pair[0]
    label = pair[1]
    if (label == 1):
        if (avg_brightness(image) < min_day_brightness):
            min_day_brightness = avg_brightness(image)
        if (avg_brightness(image) > max_day_brightness):
            max_day_brightness = avg_brightness(image)
        if (avg_hue(image) < min_day_hue):
            min_day_hue = avg_hue(image)
        if (avg_hue(image) > max_day_hue):
            max_day_hue = avg_hue(image)
    else:
        if (avg_brightness(image) > max_night_brightness):
            max_night_brightness = avg_brightness(image)
        if (avg_brightness(image) < min_night_brightness):
            min_night_brightness = avg_brightness(image)
        if (avg_hue(image) > max_night_hue):
            max_night_hue = avg_hue(image)
        if (avg_hue(image) < min_night_hue):
            min_night_hue = avg_hue(image)

threshold = (max_night_brightness + min_day_brightness) / 2

print("Threshold should be: {}".format(threshold))

random.shuffle(standardized_testing_images)

brightness_threshold = 100
hue_threshold = 20

CLASSIFIED, MISCLASSIFIED = get_classified_images_brightness_hue(standardized_testing_images, brightness_threshold, hue_threshold)

accuracy = (len(standardized_testing_images) - len(MISCLASSIFIED)) / len(standardized_testing_images)

print('Accuracy: {}'.format(accuracy))
print('Number of misclassified images = {} out of {}'.format(str(len(MISCLASSIFIED)), str(len(standardized_testing_images))))

#for image in misclassified_images:
#    if (image[2] == 1):
#        print("Day image had avg brightness: {} | predicted: {} | correct: {}".format(avg_brightness(image[0]), image[1], image[2]))
#        plt.imshow(image[0])
#        plt.show()

#for image in misclassified_images:
#    if (image[2] == 0):
#        print("Night image had avg brightness: {} | predicted: {} | correct: {}".format(avg_brightness(image[0]), image[1], image[2]))
#        plt.imshow(image[0])
#        plt.show()

#for index,image in enumerate(misclassified_images):
#    plot_HSV_channels(image[0], index)
#plt.show()

#image_num = 0
#test_im = standardized_training_images[image_num][0]
#test_label = standardized_training_images[image_num][1]

max_night_brightness_miss = 0
min_night_brightness_miss = 255
min_day_brightness_miss = 255
max_day_brightness_miss = 0
max_night_hue_miss = 0
min_night_hue_miss = 255
min_day_hue_miss = 255
max_day_hue_miss = 0

for trio in MISCLASSIFIED:
    image = trio[0]
    label = trio[2]
    if (label == 1):
        if (avg_brightness(image) < min_day_brightness_miss):
            min_day_brightness_miss = avg_brightness(image)
        if (avg_brightness(image) > max_day_brightness_miss):
            max_day_brightness_miss = avg_brightness(image)
        if (avg_hue(image) < min_day_hue_miss):
            min_day_hue_miss = avg_hue(image)
        if (avg_hue(image) > max_day_hue_miss):
            max_day_hue_miss = avg_hue(image)
    else:
        if (avg_brightness(image) > max_night_brightness_miss):
            max_night_brightness_miss = avg_brightness(image)
        if (avg_brightness(image) < min_night_brightness_miss):
            min_night_brightness_miss = avg_brightness(image)
        if (avg_hue(image) > max_night_hue_miss):
            max_night_hue_miss = avg_hue(image)
        if (avg_hue(image) < min_night_hue_miss):
            min_night_hue_miss = avg_hue(image)

max_night_brightness_true = 0
min_night_brightness_true = 255
min_day_brightness_true = 255
max_day_brightness_true = 0
max_night_hue_true = 0
min_night_hue_true = 255
min_day_hue_true = 255
max_day_hue_true = 0

for trio in CLASSIFIED:
    image = trio[0]
    label = trio[2]
    if (label == 1):
        if (avg_brightness(image) < min_day_brightness_true):
            min_day_brightness_true = avg_brightness(image)
        if (avg_brightness(image) > max_day_brightness_true):
            max_day_brightness_true = avg_brightness(image)
        if (avg_hue(image) < min_day_hue_true):
            min_day_hue_true = avg_hue(image)
        if (avg_hue(image) > max_day_hue_true):
            max_day_hue_true = avg_hue(image)
    else:
        if (avg_brightness(image) > max_night_brightness_true):
            max_night_brightness_true = avg_brightness(image)
        if (avg_brightness(image) < min_night_brightness_true):
            min_night_brightness_true = avg_brightness(image)
        if (avg_hue(image) > max_night_hue_true):
            max_night_hue_true = avg_hue(image)
        if (avg_hue(image) < min_night_hue_true):
            min_night_hue_true = avg_hue(image)

avg_brightness_day = 0
avg_hue_day = 0
avg_brightness_true_day = 0
avg_hue_true_day = 0
avg_brightness_miss_day = 0
avg_hue_miss_day = 0
avg_brightness_night = 0
avg_hue_night = 0
avg_brightness_true_night = 0
avg_hue_true_night = 0
avg_brightness_miss_night = 0
avg_hue_miss_night = 0

num_day = 0
num_night = 0
num_day_true = 0
num_night_true = 0
num_day_false = 0
num_night_false = 0

for trio in CLASSIFIED:
    image = trio[0]
    label = trio[2]
    if (label == 1):
        avg_brightness_true_day += avg_brightness(image)
        avg_hue_true_day += avg_hue(image)
        num_day_true += 1
    else:
        avg_brightness_true_night += avg_brightness(image)
        avg_hue_true_night += avg_hue(image)
        num_night_true += 1
        
for trio in MISCLASSIFIED:
    image = trio[0]
    label = trio[2]
    if (label == 1):
        avg_brightness_miss_day += avg_brightness(image)
        avg_hue_miss_day += avg_hue(image)
        num_day_false += 1
    else:
        avg_brightness_miss_night += avg_brightness(image)
        avg_hue_miss_night += avg_hue(image)
        num_night_false += 1

for pair in standardized_testing_images:
    image = pair[0]
    label = pair[1]
    if (label == 1):
        avg_brightness_day += avg_brightness(image)
        avg_hue_day += avg_hue(image)
        num_day += 1
    else:
        avg_brightness_night += avg_brightness(image)
        avg_hue_night += avg_hue(image)
        num_night += 1

avg_brightness_day /= num_day
avg_hue_day /= num_day
avg_brightness_true_day /= num_day_true
avg_hue_true_day /= num_day_true
avg_brightness_miss_day /= num_day_false
avg_hue_miss_day /= num_day_false
avg_brightness_night /= num_night
avg_hue_night /= num_night
avg_brightness_true_night /= num_night_true
avg_hue_true_night /= num_night_true
avg_brightness_miss_night /= num_night_false
avg_hue_miss_night /= num_night_false

#print("Night brightness: {}-{}".format(min_night_brightness, max_night_brightness))
#print("Day brightness: {}-{}".format(min_day_brightness, max_day_brightness))
#print("Night hue: {}-{}".format(min_night_hue, max_night_hue))
#print("Day hue: {}-{}".format(min_day_hue, max_day_hue))
#print("Night brightness true: {}-{}".format(min_night_brightness_true, max_night_brightness_true))
#print("Day brightness true: {}-{}".format(min_day_brightness_true, max_day_brightness_true))
#print("Night hue true: {}-{}".format(min_night_hue_true, max_night_hue_true))
#print("Day hue true: {}-{}".format(min_day_hue_true, max_day_hue_true))
#print("Night brightness miss: {}-{}".format(min_night_brightness_miss, max_night_brightness_miss))
#print("Day brightness miss: {}-{}".format(min_day_brightness_miss, max_day_brightness_miss))
#print("Night hue miss: {}-{}".format(min_night_hue_miss, max_night_hue_miss))
#print("Day hue miss: {}-{}".format(min_day_hue_miss, max_day_hue_miss))

print("Night brightness: {} | true: {} | false: {}".format(avg_brightness_night, avg_brightness_true_night, avg_brightness_miss_night))
print("Day brightness: {} | true: {} | false: {}".format(avg_brightness_day, avg_brightness_true_day, avg_brightness_miss_day))
print("Night hue: {} | true: {} | false: {}".format(avg_hue_night, avg_hue_true_night, avg_hue_miss_night))
print("Day hue: {} | true: {} | false: {}".format(avg_hue_day, avg_hue_true_day, avg_hue_miss_day))