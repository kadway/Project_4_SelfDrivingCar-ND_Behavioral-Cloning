import os
import csv
import cv2
import numpy as np
from model import generator
# sample data:
#  wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
# dropbox saved data from simulator:
# wget --max-redirect=20 -O data.zip https://www.dropbox.com/s/jq5ucv0xb41st57/data.zip?dl=0
# unzip data.zip

#csv_path = '../../opt/my_driving_log.csv'
#csv_path = '../../opt/Udacity_driving_log.csv'
#csv_path = '../../opt/combined_data.csv'
csv_path = '../../opt/driving_log.csv'
samples = []
with open(csv_path) as csvfile:
    reader = csv.reader(csvfile)
    for idx, line in enumerate(reader):
        samples.append(line)
#samples.pop(0)
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
batch_size = 150
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Preprocess incoming data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3), name='lambda'))
model.add(Cropping2D(cropping=((70,25), (0,0)), name='crop'))
model.add(Conv2D(24, (5, 5), subsample=(2,2), activation='relu',name='conv1'))
model.add(Conv2D(36, (5, 5), subsample=(2,2), activation='relu', name='conv2'))
model.add(Conv2D(48, (5, 5), subsample=(2,2), activation='relu', name='conv3'))
model.add(Conv2D(64, (3, 3),activation='relu', name='conv4'))
model.add(Conv2D(64, (3, 3),activation='relu', name='conv5'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history_object=model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size, epochs=5, verbose = 1)
#model_name = "model_mix_data.h5"
#model_name = "model_Udacity_data.h5"
model_name = "model_my_data.h5"
model.save(model_name)

#get some images for visualization of the CNN
images = []
for idx, line in enumerate(validation_samples):
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../../opt/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    if idx > 9:
        break
  
input_images = np.asarray(images)

from keras.models import Model
norm = Model(model.input, model.get_layer('lambda').output)
crop = Model(model.input, model.get_layer('crop').output)
convolution1 = Model(model.input, model.get_layer('conv1').output)
convolution2 = Model(model.input, model.get_layer('conv2').output)
convolution3 = Model(model.input, model.get_layer('conv3').output)
convolution4 = Model(model.input, model.get_layer('conv4').output)
convolution5 = Model(model.input, model.get_layer('conv5').output)

out_norm  = norm.predict(input_images)
out_crop  = crop.predict(input_images)
out_conv1 = convolution1.predict(input_images)
out_conv2 = convolution2.predict(input_images)
out_conv3 = convolution3.predict(input_images)
out_conv4 = convolution4.predict(input_images)
out_conv5 = convolution5.predict(input_images)

for idx,image in enumerate(out_norm):
    cv2.imwrite("images/lambda{}.jpg".format(idx), image)
for idx,image in enumerate(out_crop):
    cv2.imwrite("images/crop{}.jpg".format(idx),image)
for idx,image in enumerate(out_conv1):
    cv2.imwrite("images/conv1_{}.jpg".format(idx),image)
for idx,image in enumerate(out_conv2):
    cv2.imwrite("images/conv2_{}.jpg".format(idx),image)
for idx,image in enumerate(out_conv3):
    cv2.imwrite("images/conv3_{}.jpg".format(idx),image)
for idx, image in enumerate(out_conv4):
    cv2.imwrite("images/conv4_{}.jpg".format(idx), image)
for idx, image in enumerate(out_conv5):
    cv2.imwrite("images/conv5_{}.jpg".format(idx), image)

import matplotlib.pyplot as plt
### print the keys contained in the history object
print(history_object.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss. batch_size {}'.format(batch_size))
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')

i = 0
while os.path.exists("plots/{}_Loss{}.jpg".format(model_name,i)):
    i += 1
plt.savefig("plots/{}_Loss{}.jpg".format(model_name,i))



    
    
    
                                 
      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  