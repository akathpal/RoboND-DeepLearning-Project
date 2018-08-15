[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)




## Deep Learning Follow Me Project Solution ##

In this project, I have trained a deep neural network to identify and follow a target in Unity Quadrotor simulator shown in Fig below. I have achieved accuracy of 44%(even without collecting any additional data) or Intersect over Union(IOU) score of 0.44.

## Setup Instructions

**Dependencies**
Download the dependencies mentioned in ReadMe.md

**Clone the repository**
```
$ git clone https://github.com/akathpal/RoboND-DeepLearning-Project.git
```

**Download the data**
```
$ cd RoboND-DeepLearning-Project/data
$ ./run_me

Select 1 for downloading my training data and 2 for udacity training data.
```

## Collecting Training Data 

Although, I was getting accuracy more than 40% but I wanted to collect additional data because that is also an important step of implementing deep learning project. So, I choose the approach for collecting more data mentioned in the lectures such as zig-zag motion (for capturing view of target from all directions), target in a cluttered environment with more people and getting images while on patrol. All these scenes with the patrol points, hero path and people spawn points are shown in figures below.

[image_0]: ./docs/misc/train1.png
![alt text][image_0] 

[image_1]: ./docs/misc/train2.png
![alt text][image_1]
 
[image_2]: ./docs/misc/train3.png
![alt text][image_2] 

## Image Preprocessing

The next step after collecting the data is to preprocess the data which converts the depth masks into binary masks suitable for training the neural network. In our case, we dont require any depth information of how far the target is in the image. Our goal is to just detect if there are pixels similar to target in the image or not. 
Image preprocess function also converts png to jpeg format for reducing the size of the dataset.

## FCN Architecture

The next step is to make a fully concolution network(FCN) model. FCN model consists of mainly three parts: encoder, 1x1 convolution layer and decoder layer. I tried different architecture with more layers and more no. of filters but as total no. of parameters kept increasing. The training time increased a lot even when using AWS GPU instance. So, In the end I finally decided to keep the network only 2 encoder layer, 1conv layer, and 2 decoder layers for this follow-me project.

### Network Achitecture Diagram
[image]: ./docs/misc/architecture_model.png
![alt text][image] 

#### Encoder

Encoder consists of Separable Convolution layers followed by batch normalization. Separable convolution layer comprise of a convolution performed over each channel of an input layer which is followed by a 1x1 convolution.
I tried commenting the batch normalization and observed the results. I had to lower the learning rate but i still didn't get satisfactory results. So, batch normalization is very important to train the network faster and allow higher learning rate.These layers extract the features from the images.
I have used 64 and 128 for filters.

#### 1x1 Convolution
This is very important layer. It makes the model deeper and have more parameters. These are also very less computationally intensive because they are just matrix multiplications. These concolutions require a kernel and stride of 1. I have used filters of depth 256.

Why 1x1 Convolution Layer is used over fully-connected layer?

Our goal in this project is to determine where is the target in the image, so that we can follow it. We want to preserve spatial information. Fully Connected layers don't preserve spatial information. On the other hand, convolution layers preserve spatial information. One more advantage of using convolution layer instead of full-connected is that we can give inputs of any size. So, this is the main reason of using convolutional layers instead of fully-connected layers. The output tensor of convolution layer is 4D as you can see from model summary where as the fully-connected layers, the output flattens and no spatial information of pixels is there.

#### Decoder
Decoder consists of bilinear upsampling layer, concatenation layer and 2 separable convolution layers. Decoder block scales the low resolution to original resolution as same as input, this helps network to classify all the pixels. To recover the spatial information properly, skip connections are used. Output of pooling layer from encoders is connected to decoder layers. This helps network to use multiple resolution data for recovering spatial information. This is done by concatenate function.

### Model Summary

|Layer (type)            |  Output Shape       |     Param #|   
| :----------------------|:-------------------:| ----------:|
|input_1 (InputLayer)    | (None, 160, 160, 3) |       0    |    
|separable_conv2d_keras_1| (None, 80, 80, 64)  |       283  |    
|batch_normalization_1   |(None, 80, 80, 64)   |     256    |   
|separable_conv2d_keras_2|(None, 40, 40, 128)  |8896        |
|batch_normalization_2   |(None, 40, 40, 128)  | 512        |       
|conv2d_1 (Conv2D)       |  (None, 40, 40, 256)|       33024|     
|batch_normalization_3   |(None, 40, 40, 256)  |       1024 |     
| bilinear_up_sampling2d_1 |(None, 80, 80, 256)  |     0      |   
|concatenate_1            | (None, 80, 80, 320) |     0      |   
|separable_conv2d_keras_3 |(None, 80, 80, 128)  |    43968   |  
|batch_normalization_4    |(None, 80, 80, 128)  |    512     |  
|separable_conv2d_keras_4 |(None, 80, 80, 128)  |    17664   |  
|batch_normalization_5    |(None, 80, 80, 128)  |    512     |  
|bilinear_up_sampling2d_2 |(None, 160, 160, 128)|    0       |  
|concatenate_2            |(None, 160, 160, 131)|     0      |   
|separable_conv2d_keras_5 |(None, 160, 160, 64) |     9627   |  
|batch_normalization_6    |(None, 160, 160, 64) |    256     |  
|separable_conv2d_keras_6 | (None, 160, 160, 64)|     4736   |    
|batch_normalization_7    |(None, 160, 160, 64) |    256     |  
|conv2d_2 (Conv2D)        |  (None, 160, 160, 3)|      195   |    

=================================================================

Total params: 121,721
Trainable params: 120,057
Non-trainable params: 1,664

## Hyper Parameters

### Learning Rate
I kept my learning rate at 0.005. I experimented with mltiple values. By increasing the learning rate more that 0.01, I was getting a very low accuracy and a big variation between valiation loss and training loss.
By decreasing the learning rate below 0.0005, the training loss and validation loss cannot converge.
Good fit of model is represented by training loss less than validation loss and validation loss almost equal to training loss. This type of fit I achieved with learning rate 0.001 and 0.005. But I decided to keep it at 0.005 because of slight improvemnets.

### Batch Size
Batch size is number of samples that get propagated in one pass. It should be maximum that your GPU can handle. So, I tried using 32 and 64. Got better results with 64.

### No. of epochs
This is an important factor which can lead to underfit or overfit the model. It also depends on the convergence of training loss and validation loss. In my case, I kept it at 15.

### Steps per Epochs
Recommended value for this is no. of training images divided by batch size. I kept my value at 100 slightly more than that.

### Validation Steps per Epoch
Validation step is similar to steps per epoch but they are for validation data. This is number of batches of validation images that go through network in 1 epoch, I kept this value at 50.

### Workers
I tried experimenting by changing no. of workers from 2 to 4. I noticed a littlr bit of speed improvement but not that significant. In the end, I kept no. of workers(spinning multiple number of processes) as 4.

## Improvements

When I ran the simulator using my model. The quadrotor sometimes fail to detect hero in some frames but once detected it follows the hero properly. I need to collect more data in which hero is far from quadrotor but in the frame of camera. This will further improve the accuracy of model. But colecting this data and training using that is very computational intensive and time-consuming process.  

For different scenarios, for ex to detect dog, we have to collect additional data for dog or anything else from all the angles, from far away , from nearby , in cluttered environment with dogs and humans. Then accordingly tune the hyper parameters again. So the same model that we trained wont work because we need to tune the parameters and might need to build a more deeper model. 


## Experimentation: Testing in Simulation

Run the realtime follower script
```
$ python follower.py model_weights.h5
```

The output of my final testing in simulator is shown below:

[image_3]: ./docs/misc/follow.png
![alt text][image_3]

