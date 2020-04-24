#!/usr/bin/env python3

# Implementation of Oriented FAST and Rotated BRIEF (ORB) Algorithm for Face Detection

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Set the default figure size
plt.rcParams['figure.figsize'] = [20,10]

# Load the training image
image_training = cv2.imread('./images/face.jpeg')
image_query = cv2.imread('./images/face.jpeg')
image_small = cv2.imread('./images/faceQS.png')
image_rotate = cv2.imread('./images/faceR.jpeg')
image_bright = cv2.imread('./images/faceRI.png')
image_noise = cv2.imread('./images/faceRN5.png')
image_test = cv2.imread('./images/Team.jpeg')

# Convert images to gray scale
training_image = cv2.cvtColor(image_training, cv2.COLOR_BGR2RGB)
query_image = cv2.cvtColor(image_query, cv2.COLOR_BGR2RGB)
small_image = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
rotate_image = cv2.cvtColor(image_rotate, cv2.COLOR_BGR2RGB)
bright_image = cv2.cvtColor(image_bright, cv2.COLOR_BGR2RGB)
noise_image = cv2.cvtColor(image_noise, cv2.COLOR_BGR2RGB)
test_image = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)

training_gray = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)
query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
small_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
rotate_gray = cv2.cvtColor(rotate_image, cv2.COLOR_BGR2GRAY)
bright_gray = cv2.cvtColor(bright_image, cv2.COLOR_BGR2GRAY)
noise_gray = cv2.cvtColor(noise_image, cv2.COLOR_BGR2GRAY)
test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# set ORB parameters
nfeatures = 5000            # max number of keypoints  (x)
scaleFactor = 2.0           # pyramid decimation ratio (x)
nlevels = 8                 # number of pyramid levels
edgeThreshold = 31          # size of border where features not detected
firstLevel = 0              # which level in pyramid treated as first level
WTA_K = 2                   # number of random pixels to produce each element of BRIEF descriptor (2,3,or 4), index or brightest pixel returned
patchSize = 31              # size of patch used by oriented BRIEF descriptor
fastThreshold = 20          
orb = cv2.ORB_create(   nfeatures=nfeatures,
                        scaleFactor=scaleFactor,
                        nlevels=nlevels,
                        edgeThreshold=edgeThreshold,
                        firstLevel=firstLevel,
                        WTA_K=WTA_K,
                        patchSize=patchSize,
                        fastThreshold=fastThreshold)

# find keypoints and compute descriptor in images (None = not using mask)
keypoints_train, descriptor_train = orb.detectAndCompute(training_gray, None)
keypoints_query, descriptor_query = orb.detectAndCompute(query_gray, None)
keypoints_small, descriptor_small = orb.detectAndCompute(small_gray, None)
keypoints_rotate, descriptor_rotate = orb.detectAndCompute(rotate_gray, None)
keypoints_bright, descriptor_bright = orb.detectAndCompute(bright_gray, None)
keypoints_noise, descriptor_noise = orb.detectAndCompute(noise_gray, None)
keypoints_test, descriptor_test = orb.detectAndCompute(test_gray, None)

# set Brute-Force keypoint matching parameters 
normType = cv2.NORM_HAMMING     # metric used to determine quality of match
crossCheck = True               # enable double checking of match in reverse
bf = cv2.BFMatcher( normType=normType, crossCheck=crossCheck)

# perform matching between ORB descriptors using Brute-Force
matches_query = bf.match(descriptor_train, descriptor_query)
matches_small = bf.match(descriptor_train, descriptor_small) 
matches_rotate = bf.match(descriptor_train, descriptor_rotate) 
matches_bright = bf.match(descriptor_train, descriptor_bright) 
matches_noise = bf.match(descriptor_train, descriptor_noise) 
matches_test = bf.match(descriptor_train, descriptor_test) 

# draw best 300 matches
matches_query = sorted(matches_query, key = lambda x : x.distance)
result_query = cv2.drawMatches(training_gray, keypoints_train, query_gray, keypoints_query, matches_query[:300], query_gray, flags = 2)
matches_small = sorted(matches_small, key = lambda x : x.distance)
result_small = cv2.drawMatches(training_gray, keypoints_train, small_gray, keypoints_small, matches_small[:300], small_gray, flags = 2)
matches_rotate = sorted(matches_rotate, key = lambda x : x.distance)
result_rotate = cv2.drawMatches(training_gray, keypoints_train, rotate_gray, keypoints_rotate, matches_rotate[:300], rotate_gray, flags = 2)
matches_bright = sorted(matches_bright, key = lambda x : x.distance)
result_bright = cv2.drawMatches(training_gray, keypoints_train, bright_gray, keypoints_bright, matches_bright[:300], bright_gray, flags = 2)
matches_noise = sorted(matches_noise, key = lambda x : x.distance)
result_noise = cv2.drawMatches(training_gray, keypoints_train, noise_gray, keypoints_noise, matches_noise[:300], noise_gray, flags = 2)
matches_test = sorted(matches_test, key = lambda x : x.distance)
result_test = cv2.drawMatches(training_gray, keypoints_train, test_gray, keypoints_test, matches_test[:300], test_gray, flags = 2)

# Print the number of keypoints detected in the images
print("Number of Keypoints Detected In The Training Image: ", len(keypoints_train))
print("Number of Keypoints Detected In The Query Image: ", len(keypoints_query))
print("Number of Keypoints Detected In The Small Image: ", len(keypoints_small))
print("Number of Keypoints Detected In The Rotated Image: ", len(keypoints_rotate))
print("Number of Keypoints Detected In The Bright Image: ", len(keypoints_bright))
print("Number of Keypoints Detected In The Noisy Image: ", len(keypoints_noise))
print("Number of Keypoints Detected In The Test Image: ", len(keypoints_test))

# Print total number of matching points between the training and other images
print("Number of Matching Keypoints Between The Training and Query Image: ", len(matches_query))
print("Number of Matching Keypoints Between The Training and Small Image: ", len(matches_small))
print("Number of Matching Keypoints Between The Training and Rotated Image: ", len(matches_rotate))
print("Number of Matching Keypoints Between The Training and Bright Image: ", len(matches_bright))
print("Number of Matching Keypoints Between The Training and Noisy Image: ", len(matches_noise))
print("Number of Matching Keypoints Between The Training and Test Image: ", len(matches_test))

# draw keypoints without size and orientation on training image
#keyp_without_size = np.copy(training_image)
#cv2.drawKeypoints(training_image, keypoints_train, keyp_without_size, color = (0,255,0))
# draw keypoints with size and orientation on training image
#keyp_with_size = np.copy(training_image)
#cv2.drawKeypoints(training_image, keypoints_train, keyp_with_size, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

f,x = plt.subplots(2,6)
x[0,0].imshow(query_image), x[0,0].set_title('Image')
x[1,0].imshow(result_query), x[1,0].set_title('Same Image Matches')
x[0,1].imshow(small_image), x[0,1].set_title('Small Image')
x[1,1].imshow(result_small), x[1,1].set_title('Small Image Matches')
x[0,2].imshow(rotate_image), x[0,2].set_title('Rotate Image')
x[1,2].imshow(result_rotate), x[1,2].set_title('Rotate Image Matches')
x[0,3].imshow(bright_image), x[0,3].set_title('Bright Image')
x[1,3].imshow(result_bright), x[1,3].set_title('Bright Image Matches')
x[0,4].imshow(noise_image), x[0,4].set_title('Noisy Image')
x[1,4].imshow(result_noise), x[1,4].set_title('Noisy Image Matches')
x[0,5].imshow(test_image), x[0,5].set_title('Test Image')
x[1,5].imshow(result_test), x[1,5].set_title('Test Image Matches')
plt.show()