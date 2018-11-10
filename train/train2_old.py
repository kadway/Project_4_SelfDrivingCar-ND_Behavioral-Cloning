import os
import csv
import cv2
import numpy as np
# dropbox saved data from simulator:
# wget --max-redirect=20 -O track1.zip https://www.dropbox.com/sh/gyk9kunpltqpp2f/AAAFdcA3mzlknUcLG5BIpJWRa?dl=0
# unzip track1.zip && tar -xf data_track1.tar.gz

lines = []
with open('../../opt/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
 
car_images = []
steering_angles = []
img_path = '../../opt/IMG/'

for line in lines:    
    # get the image names from the read in line
    img_center_name = line[0]
    img_center_name = img_center_name.split('/')[-1]
    img_left_name   = line[1]
    img_left_name   = img_left_name.split('/')[-1]
    img_right_name  = line[2]
    img_right_name  = img_right_name.split('/')[-1]
   
    # read in images from center, left and right cameras and mirror center image
    img_center = cv2.imread(img_path + img_center_name)
    img_left  = cv2.imread(img_path + img_left_name)
    img_right = cv2.imread(img_path + img_right_name)
    # mirror center image to get more generalized data
    img_center_mirror = cv2.flip(img_center,1)
    
    #get steering value from the read in line
    steering_center = float(line[3])
    #invert steering sign for mirror center image
    steering_center_mirror = steering_center*-1.0
    
    # create adjusted steering measurements for the side camera images
    correction = 0.1 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
            
    # add images and angles to data set
    car_images.extend((img_center, img_left, img_right, img_center_mirror))
    steering_angles.extend((steering_center, steering_left, steering_right, steering_center_mirror))
            
X_train = np.array(car_images)
y_train = np.array(steering_angles)

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(10, (5, 5),activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, (5, 5),activation='relu', padding='valid'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=8)
model.save('model2.h5')

      

    
    
    
                                 
      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  