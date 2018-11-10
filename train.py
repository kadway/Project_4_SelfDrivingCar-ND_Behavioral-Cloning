import os
import csv
import cv2
import numpy as np
from model import generator

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
    
# sample data:
#  wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

# dropbox saved data from simulator:
# wget --max-redirect=20 -O data.zip https://www.dropbox.com/s/jq5ucv0xb41st57/data.zip?dl=0

def train_cnn(model_name = "model.h5"):

    #csv_path = 'train_data/driving_log.csv'
    csv_path = 'train_data/driving_log_2tracks.csv'
    samples = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for idx, line in enumerate(reader):
            samples.append(line)

    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    batch_size = 150
    train_generator = generator(train_samples, batch_size)
    validation_generator = generator(validation_samples, batch_size)
    
    model = Sequential()
    
    # Preprocess incoming data
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3), name='lambda'))
    model.add(Cropping2D(cropping=((50,20), (0,0)), name='crop'))
    
    #add convolutional layers
    model.add(Conv2D(24, (5, 5), subsample=(2,2), activation='relu',name='conv1'))
    model.add(Conv2D(36, (5, 5), subsample=(2,2), activation='relu', name='conv2'))
    model.add(Conv2D(48, (5, 5), subsample=(2,2), activation='relu', name='conv3'))
    model.add(Conv2D(64, (3, 3),activation='relu', name='conv4'))
    model.add(Conv2D(64, (3, 3),activation='relu', name='conv5'))
    
    #add dropout layer and then flatten before fully connected layers
    model.add(Dropout(0.5))
    model.add(Flatten())
    
    #add fully connected layers
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    #network training
    model.compile(loss='mse', optimizer='adam')
    history_object=model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size, epochs=5, verbose = 1)
    model.save(model_name)
    
    return history_object

def plot_loss(history_object, model_name = "model.h5"):

    ### print the keys contained in the history object
    print(history_object.history.keys())
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss.')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')

    i = 0
    while os.path.exists("plots/{}_Loss{}.jpg".format(model_name,i)):
        i += 1
    plt.savefig("plots/{}_Loss{}.jpg".format(model_name,i))
    
    return
    
                                 
      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  