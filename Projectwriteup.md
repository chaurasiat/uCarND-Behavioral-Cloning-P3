# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md  summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

For the purpose of the project I have used the architecture published by the Team at Nvidia 
Here's the architecture:

![alt text](WriteupImages/nvidia_architecture.png)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 121,126). 

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model uses an adam optimizer, and the default learning rate is used .Batch Size=32,Epochs=3,correction factor for steering=0.2

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I didnt tested with my own data as during driving,I realized I was not good at driving with keyboard,so i decided to go with the data given by the udacity

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try out various network architectures  and to  find out how well it performed on the track provided by the simulator 

My first step was to use a convolution neural network model similar to the comma.ai's architecture:
https://arxiv.org/pdf/1608.01230.pdf
 I thought this model might be appropriate because it worked pretty well with the given problem but the validation error increased with epoch,and also on re running model.py,it was giving different result like one model.h5 was running perfectly fine while other,it went into water. To combat the overfitting,I modified the model so that it used Dropouts of 0.2 on the flattened layer and 0.5 on the first Dense layers. Then I changed the activation on the dense layer to elu to avoid any dead nodes.**model.h5 and run1.mp4 are the output for  comma ai architecture** 
So afterwards,i used a convolution neural network model similar to the NVIDIA architecture,
https://arxiv.org/pdf/1604.07316v1.pdf
I modified NVIDIA architecture to reduce overfitting issue,added 0.5 in convolutional layer and 0.2 in fully connected layer.
**nvidiamodel1.h5 and nvidiaRun.mp4 are the output with NVIDIA architecture**

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I adjusted the steering correction for the left and right images to a correction factor of about 0.2

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Output:

![alt text](WriteupImages/nvidiaOutput1.png)

#### 2. Final Model Architecture

The final model architecture (model.py lines 114-123) consisted of a convolution neural network with the following layers and layer sizes ...

![alt text](WriteupImages/nvidiaVis.png)





#### 3. Creation of the Training Set & Training Process
I didnt tested with my own data as during driving,I realized I was not good at driving with keyboard,so i decided to go with the data given by the udacity

I have done **data Augmentation** on the above , to drastically change the size of data by a factor of 6  In the generator function I have used the images from all three cameras also, I am adding a correction factor of 0.2 to the steering angle for the images from left and right cameras,

Center left and right Images :

![alt text](WriteupImages/center_2016_12_01_13_30_48_287.jpg) ![alt text](WriteupImages/left_2016_12_01_13_30_48_287.jpg)


![alt text](WriteupImages/right_2016_12_01_13_30_48_287.jpg)

 After this , I flipped these images  and also I  added a corresponding steering angle multiplied by a factor of  -1.I :

Original and Flipped Image:

![alt text](WriteupImages/center_2016_12_01_13_30_48_287.jpg) 
  
 ![alt text](WriteupImages/center_2016_12_01_13_30_48_287_horizontal.jpg)
 
 
 Then preprocessed this data by by doing Lambda normalization and Cropping, which have been  discussed in the classroom,this helped the most in training the network more efficiently.

The data has been shuffled and 20% of data has been split to validation data set.

I used the augmented training data for training the model. The validation set helped determine if the model was over or under fitting. The final number of epochs that i went with was 3(I arrived at this number by numerous  hit and trials). I used an adam optimizer so that manually training the learning rate wasn't necessary.

 


