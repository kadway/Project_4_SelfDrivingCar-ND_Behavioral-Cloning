# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./CNN_images/output/lambda4.jpg "lambda layer output"
[image2]: ./CNN_images/output/crop4.jpg "crop layer output"
[image3]: ./CNN_images/output/conv1.jpg "conv1 layer output"
[image4]: ./plots/model_loss.jpg "model loss"
[image5]: ./plots/model_b105e10_loss0.jpg "model loss 1st"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* video.py to make a video from the autonomous driving simulation which outputs image frames
* visualization.py contains code to visualize the CNN layers output
* model.h5 containing a trained convolution neural network 
* video.mp4 a video from the autonomous driving of the first track predicted by model.h5
* video_track2.mp4 a video from the autonomous driving of the second track predicted by model.h5
* Behavioral_Cloning.html is the html of a Jupyter notebook containing the steps performed to train the first working model
* Behavioral_Cloning2.html is the html of a Jupyter notebook containing the steps performed to train the final working model
* writeup_report.md summarizing the results


#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network (CNN) based on the architecture presented in NVIDIAs paper
from 25 April 2016 `End to End Learning for Self-Driving Cars`.
It has a lambda layer for image normalization, a crop layer for image cropping, 5 convolutional layers,
1 dropout layer, a flattening layer and 3 fully connected layers like shown bellow:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
|================================================================
lambda (Lambda)              (None, 160, 320, 3)       0         
_________________________________________________________________
crop (Cropping2D)            (None, 90, 320, 3)        0         
_________________________________________________________________
conv1 (Conv2D)               (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2 (Conv2D)               (None, 20, 77, 36)        21636     
_________________________________________________________________
conv3 (Conv2D)               (None, 8, 37, 48)         43248     
_________________________________________________________________
conv4 (Conv2D)               (None, 6, 35, 64)         27712     
_________________________________________________________________
conv5 (Conv2D)               (None, 4, 33, 64)         36928     
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
|================================================================

The Jupyter html file can be seen [here](Behavioral_Cloning.html), where the training steps can be visualized.
The training and validation loss of the first model is shown bellow:

![alt text][image5] 
    
The code where the CNN is defined be found on `model.py`, in function `train_cnn(epochs_in, batch_size = 120, model_name = "model.h5")` 

The model includes RELU activations to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 106). 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py lines 117). 

The model was trained and validated on different data sets to ensure that the model was not overfitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 127).
I trained for 10 epochs with a batch size of 128 samples, the dropout layer has a keep probability of 50%.


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I used the sample training data provided by Udacity as a starting point. 
The training data contained images from center, left and right cameras.
I also obtained training images from driving the second track on the Udacity's simulator.
Additionally, instead to driving the track on the opposite direction to generalize the model,
I added flipped images of the center camera images and added also their respective inverted steering angle.
This helped to generalize the model and allow the car also to drive, at least for a while, in the second track.
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the NVIDIAs design as a base and then tune it
to get the car to stay on the road.
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

Because the first attempts to train the model were very memory consuming,
I opted to use a generator (line 36 in model.py) to load the images as they were necessary.
The generator receives as input the image paths and the steering angles from the center camera images.
Inside the generator function I read the images in batches, and created a list of images and a list with the respective steering angles.
For each center camera image I created an additional flipped image, with the inverted steering angle.
The simulator only provides steering angles for the center camera images and therefore the steering angles for the left and right camera 
images had to be calculated (lines 66-68 in model.py).

An attempt to have the flipped side camera images in the training data was done but did not prove helpful as the car was still driving off road.
So I proceeded by flipping only the center camera images which gave a better result but still not not enough to have the car stay on the road.
I then cropped the input images after the lambda layer and added a function to increase the brightness of the input images (line 24 in model.py) before feeding 
them to the network.
This helps the network to focus on the relevant part of the images while training.
While training, the images were also shuffled from inside the generator function (line 40 in model.py).
To combat the overfitting, I modified the model and introduced a dropout layer with keep probability of 50%.

The next step was to run the simulator to see how well the car was driving around track one.

A Jupyter notebook was used to train the network and test the code.
For the first model the html `Behavioral_Cloning.html` file is provided where we can see the progress of training
for a batch of 105 samples and 10 epochs.
For this model the training loss was 0.0220 and the validation loss was 0.0219.

There were a few spots where the vehicle drove on top of the lane but still staying on the road. 
To improve the driving behavior in these cases, I increased the number of the networks's convolutional layers filters
in an attempt for the network to learn more details of the track.
At the end of the process, the vehicle is able to drive autonomously around track 1 without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 106-124) consisted of a convolution neural network with the following layers
and layer sizes:

Layer (type)                 Output Shape              Param #   
|===============================================================
lambda (Lambda)              (None, 160, 320, 3)       0         
_________________________________________________________________
crop (Cropping2D)            (None, 90, 320, 3)        0         
_________________________________________________________________
conv1 (Conv2D)               (None, 43, 158, 48)       3648      
_________________________________________________________________
conv2 (Conv2D)               (None, 20, 77, 72)        86472     
_________________________________________________________________
conv3 (Conv2D)               (None, 8, 37, 96)         172896    
_________________________________________________________________
conv4 (Conv2D)               (None, 6, 35, 128)        110720    
_________________________________________________________________
conv5 (Conv2D)               (None, 4, 33, 128)        147584    
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 33, 128)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 16896)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               1689700   
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
|=================================================================

For the final model the training loss is 0.0246 and the validation loss is 0.0230.
See the Jupyter html file [here](Behavioral_Cloning2.html) for the training steps and loss plot of this model.

The validation set helped determine if the model was over or under fitting.
The validation loss decreases consistently along the training epochs and always bellow the training loss and the ideal number of epochs was 10 as evidenced by the plot bellow:

![alt text][image4] 

I used an adam optimizer so that manually training the learning rate wasn't necessary and only the 
batch size and numbers of filters of the convolutional network layers were tuned.

#### 3. Creation of the Training Set & Training Process

The training set used was the sample training data from Udacity and was extended with additional images 
recorded from driving on second track.
See _1. Solution Design Approach_ for detailed explanation.

Example of image after normalization in the lambda layer:

![alt text][image1]

The images were cropped to remove the parts not relevant for the training model.
An example bellow:

![alt text][image2]

In `visualization.py` is an attempt to visualize the output of the convolutional layers of the network.
The visualization code is in `visualization.py` (line 49) and the function `predict_layer_output(images, model, layer_name="conv1")` 
saves the output images of a given layer as a matrix of images.
Further work would be required for a better visualization however for an idea of what the first convolutional layer output looks like,
see bellow:

![alt text][image3]  


#### 4. Simulation of autonomous driving from trained model using Udacity simulator

The resulting videos of the autonomous driving simulation using the trained model (model.h5) can be seen here for both tracks:
- [Track1](video.mp4) 
- [Track2](video_track2.mp4)

