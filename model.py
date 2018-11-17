import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import pickle
# sample data:
#  wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

# dropbox saved data from simulator:
# wget --max-redirect=20 -O data.zip https://www.dropbox.com/s/jq5ucv0xb41st57/data.zip?dl=0


img_path = 'train_data/IMG/'
#csv_path = 'train_data/driving_log.csv'
csv_path = 'train_data/driving_log_2tracks.csv'

def increase_brightness(img, brightness=255):
    #convert it to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v += brightness
    new_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)

def convert_RGB(img, brightness=255):
    img=increase_brightness(img, brightness)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
   
                # read in images from center, left and right cameras
                img_center = convert_RGB(cv2.imread(img_path + img_center_name))
                img_left  = convert_RGB(cv2.imread(img_path + img_left_name))
                img_right = convert_RGB(cv2.imread(img_path + img_right_name))
                # mirror center image to get more generalized data
                center_flipped = cv2.flip(img_center,1)  
                #left_flipped = cv2.flip(img_left,1)
                #right_flipped = cv2.flip(img_right,1)
                #get steering value
                steering_center = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                #invert steering sign for mirror center image
                steering_center_flipped = -steering_center
                #steering_left_flipped = steering_right
                #steering_right_flipped = steering_left
                # add images and angles to data set
                car_images.extend((img_center, img_left, img_right, center_flipped))
                #car_images.extend((img_center, img_left, img_right, center_flipped, left_flipped, right_flipped))
                steering_angles.extend((steering_center, steering_left, steering_right, steering_center_flipped))
                #steering_angles.extend((steering_center, steering_left, steering_right, steering_center_flipped, steering_left_flipped, steering_right_flipped))
                
            X_train = np.asarray(car_images)
            y_train = np.asarray(steering_angles)
            yield shuffle(X_train, y_train)

def train_cnn(epochs_in, batch_size = 120, model_name = "model.h5"):

    samples = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for idx, line in enumerate(reader):
            samples.append(line)

    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.20)
    
    print("Training samples: ",len(train_samples)*4)
    print("Validation samples: ", len(validation_samples)*3)
    print("")
    
    # compile and train the model using the generator function
   
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
    history_object=model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size, epochs=epochs_in, verbose = 1)
    model.save(model_name)
    name=model_name.split(".")[0]
    i = 0
    while os.path.exists("plots/{}_history{}.p".format(model, i)):
        i += 1
    with open('plots/{}_history{}.p'.format(name,i), 'wb') as f:
        pickle.dump(history_object.history, f)
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
    while os.path.exists("plots/{}_loss{}.jpg".format(model_name,i)):
        i += 1
    plt.savefig("plots/{}_loss{}.jpg".format(model_name,i))    
    return

def main():
    model_name=input("type model name like model.h5:")
    epochs_in=input("type number of epochs:")
    history=train_cnn(epochs_in, model_name)
    name=model_name.split(".")[0]
    plot_loss(history, name)

    
if __name__ == "__main__":
    # execute only if run as a script
    main()
    
    
                                 
      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  