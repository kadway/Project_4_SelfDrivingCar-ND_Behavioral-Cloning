from keras.models import Model
from keras.models import load_model
import os
import csv
import cv2
import numpy as np
from keras.utils import plot_model
import matplotlib.image as mpimg
from model import increase_brightness

csv_path = 'train_data/driving_log.csv'
img_path = 'train_data/IMG/'
def read_preview_images():
    
    images = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for idx, line in enumerate(reader):
            source_path = line[0]
            filename = source_path.split('/')[-1]
            current_path = img_path + filename
            image = increase_brightness(cv2.imread(current_path), 255)
            cv2.imwrite("CNN_images/input{}.jpg".format(idx), image)
            images.append(image)
            if idx >= 9:
                break 
    #print("Copy of input images saved to: CNN_images/input*.jpg")
    return np.asarray(images)

def show(model_name="model.h5"):
    #load the saved model
    model = load_model(model_name)
    print("Model loaded.")
    
    input_images = read_preview_images()

    norm = Model(model.input, model.get_layer('lambda').output)
    crop = Model(model.input, model.get_layer('crop').output)
    #convolution1 = Model(model.input, model.get_layer('conv1').output)
    #convolution2 = Model(model.input, model.get_layer('conv2').output)
    #convolution3 = Model(model.input, model.get_layer('conv3').output)
    #convolution4 = Model(model.input, model.get_layer('conv4').output)
    #convolution5 = Model(model.input, model.get_layer('conv5').output)

    plot_model(model, to_file='plots/model.jpg')
        
    out_norm  = norm.predict(input_images)
    print(out_norm.shape)
    out_crop  = crop.predict(input_images)
    print(out_crop.shape)
    #out_conv1 = convolution1.predict(input_images)
    #print(out_conv1.shape)
    #out_conv2 = convolution2.predict(input_images)
    #print(out_conv2.shape)
    #out_conv3 = convolution3.predict(input_images)
    #print(out_conv3.shape)
    #out_conv4 = convolution4.predict(input_images)
    #print(out_conv4.shape)
    #out_conv5 = convolution5.predict(input_images)
    #print(out_conv5.shape)
        
    for idx,image in enumerate(out_norm):
        cv2.imwrite("CNN_images/lambda{}.jpg".format(idx), image*255)
        #print("Norm layer images saved.")
    for idx,image in enumerate(out_crop):
        cv2.imwrite("CNN_images/crop{}.jpg".format(idx),image*255)
    #for idx,image in enumerate(out_conv1):
    #    for i,img in enumerate(image.shape[2])
    #    cv2.imwrite("CNN_images/conv1_{}{}.jpg".format(idx,i),img*255)
    #for idx,image in enumerate(out_conv2):
    #    cv2.imwrite("CNN_images/conv2_{}.jpg".format(idx),image)
    #for idx,image in enumerate(out_conv3):
    #    cv2.imwrite("CNN_images/conv3_{}.jpg".format(idx),image)
    #for idx, image in enumerate(out_conv4):
    #    cv2.imwrite("CNN_images/conv4_{}.jpg".format(idx), image)
    #for idx, image in enumerate(out_conv5):
    #    cv2.imwrite("CNN_images/conv5_{}.jpg".format(idx), image)
    
    return norm, out_norm


