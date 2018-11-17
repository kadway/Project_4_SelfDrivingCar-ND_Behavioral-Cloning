from keras.models import Model
from keras.models import load_model
import os
import csv
import cv2
import numpy as np
from keras.utils import plot_model
import matplotlib.image as mpimg
from model import increase_brightness
import matplotlib.pyplot as plt
from PIL import Image

img_path = "CNN_images/input/"

def read_preview_images(img_limit):
    images = []
    #get image list
    images_in = os.listdir(img_path)
    #read-in orginal image
    for fname in images_in:
        images.append(cv2.imread(img_path + fname))
    #conver to numpy array
    images_out = np.asarray(images)
    #if given limit is hight than existing images, use all the images
    if(img_limit>len(images_out)):
        img_limit=len(images_out)
    
    return images_out[:img_limit,:,:]

def show_lambda_crop_layers(input_images, model):
        
    #get the intermediate layers of the loaded model
    norm = Model(model.input, model.get_layer('lambda').output)
    crop = Model(model.input, model.get_layer('crop').output)
    #plot_model(model, to_file='plots/model.jpg')
    print("Input array shape: ", input_images.shape)
    #predict the output of the intermediate layers for the given input images
    out_norm  = norm.predict(input_images)
    print("Norm layer output shape: ", out_norm.shape)
    out_crop  = crop.predict(input_images)
    print("Crop layer output shape: ", out_crop.shape)
    for idx,image in enumerate(out_norm):
        cv2.imwrite("CNN_images/output/lambda{}.jpg".format(idx), image*255)
    for idx,image in enumerate(out_crop):
        cv2.imwrite("CNN_images/output/crop{}.jpg".format(idx),image*255)
    print("Predicted images saved.")
    return

def predict_layer_output(images, model, layer_name="lambda"):
  
    #get the intermediate layers of the loaded model
    conv_layer = Model(model.input, model.get_layer(layer_name).output)
    out_conv = conv_layer.predict(images)

    #how many filters exist
    depth=len(out_conv[0,0,0,:])
    print("The {} layer shape is {} and has {} filters".format(layer_name,out_conv.shape, depth))
    #how many input images      
    img_num = len(out_conv[:,0,0,0])
    print("There are {} input images ".format(img_num))
          
    #combine all the filters for each input image in one unique list
    out_images = []
    for image in range(0, img_num-1):
        for i in range(0, depth-1):
            out_images.append(out_conv[image,:,:,i])
    #convert to numpy array
    out_images=np.asarray(out_images)
    print("Combined filter image array has shape:", out_images.shape)
    
    width = out_images.shape[2]
    height = out_images.shape[1]
    
    #total number of pixels
    pix = img_num*depth*width*height
    # 4 columns of images
    wout = int(width*4)
    # x lines
    hout = int(pix/wout)
    print("Each filter has {} x {}".format(height,width))
    print("Combined image is {} x {} and has {} pixels".format(hout,wout, pix))
    
    #initialy empty 2d array for the images
    output = np.zeros((int(hout), int(wout)))
    
    # fill the picture with the output filters from the CNN layer
    im=0
    for i in range(0, int(wout/width)):
        for j in range(0, int((hout/height))):
            try:
                output[j*height:(j+1)*height,i*width:(i+1)*width] = out_images[im,:,:]
                im = im + 1
            except:
                break
                
    #save the resulting image
    result = Image.fromarray((output*255).astype(np.uint8))
    result.save('CNN_images/output/{}.jpg'.format(layer_name))
    
    return