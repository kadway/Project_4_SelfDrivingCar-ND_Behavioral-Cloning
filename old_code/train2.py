import os
import csv
import cv2
import numpy as np
from model import generator
# dropbox saved data from simulator:
# wget --max-redirect=20 -O track1.zip https://www.dropbox.com/sh/gyk9kunpltqpp2f/AAAFdcA3mzlknUcLG5BIpJWRa?dl=0
# unzip track1.zip && tar -xf data_track1.tar.gz

file = open('../../opt/driving_log.csv')
numlines = len(file.readlines())

samples = []
with open('../../opt/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for idx, line in enumerate(reader):
        samples.append(line)
        #print("Read line {} from {} lines".format(idx, numlines))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
batch_size = 100
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Preprocess incoming data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
#model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(70, 160, 3)))
# ch, row, col = 3, 80, 320  # Trimmed image format
# Preprocess incoming data, centered around zero with small standard deviation 
# model.add(Lambda(lambda x: x/127.5 - 1.,
#        input_shape=(ch, row, col),
#        output_shape=(ch, row, col)))

model.add(Conv2D(10, (5, 5),activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, (5, 5),activation='relu', padding='valid'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=8)
#model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, #nb_val_samples=len(validation_samples), nb_epoch=8)
model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size, epochs=5, verbose = 1)
model.save('model.h5')
    

    
    
    
                                 
      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  