import os
import cv2
import numpy as np
from sklearn.utils import shuffle

img_path = '../../opt/IMG/'

def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            car_images = []
            steering_angles = []
            for batch_sample in batch_samples:
                  # get the image names
                img_center_name = batch_sample[0]
                img_center_name = img_center_name.split('/')[-1]
                img_left_name   = batch_sample[1]
                img_left_name   = img_left_name.split('/')[-1]
                img_right_name  = batch_sample[2]
                img_right_name  = img_right_name.split('/')[-1]
   
                # read in images from center, left and right cameras and mirror center image
                img_center = cv2.imread(img_path + img_center_name)
                img_left  = cv2.imread(img_path + img_left_name)
                img_right = cv2.imread(img_path + img_right_name)
                # mirror center image to get more generalized data
               
                center_flipped = cv2.flip(img_center,1)  
                left_flipped = cv2.flip(img_left,1)
                right_flipped = cv2.flip(img_right,1)
                #get steering value from the read in line
                steering_center = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                #invert steering sign for mirror center image
                steering_center_flipped = -steering_center
                steering_left_flipped = steering_right
                steering_right_flipped = steering_left
                # add images and angles to data set
                car_images.extend((img_center, img_left, img_right, center_flipped, left_flipped, right_flipped))
                steering_angles.extend((steering_center, steering_left, steering_right, steering_center_flipped, steering_left_flipped, steering_right_flipped))
                
            X_train = np.asarray(car_images)
            y_train = np.asarray(steering_angles)
            yield shuffle(X_train, y_train)

