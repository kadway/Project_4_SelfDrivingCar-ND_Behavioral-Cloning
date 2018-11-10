import csv
import cv2
import numpy as np
# dropbox saved data from simulator:
# wget --max-redirect=20 -O track1.zip https://www.dropbox.com/sh/gyk9kunpltqpp2f/AAAFdcA3mzlknUcLG5BIpJWRa?dl=0
# unzip track1.zip && tar -xf data_track1.tar.gz
# 

file = open('../../opt/driving_log.csv')
numlines = len(file.readlines())
lines = []
with open('../../opt/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for idx, line in enumerate(reader):
        lines.append(line)
        print("Read line {} from {} lines".format(idx, numlines))
        
images = []
measurements = []
for idx, line in enumerate(lines):
    print("IMG path {} from {} lines".format(idx, numlines))
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../../opt/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for idx, (image, measurement) in enumerate(zip(images, measurements)):
    print("Augment line {} from {} lines".format(idx, numlines))
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
    
X_train = np.array(augmented_images)
print("X_train shape: ", X_train.shape)
y_train = np.array(augmented_measurements)
print("y_train shape: ", y_train.shape)

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(6, (5, 5),activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(6, (5, 5),activation='relu', padding='valid'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
model.save('model.h5')



    
    
    
                                 
      
        