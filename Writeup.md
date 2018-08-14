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

## Model Architecture

The next step is to make a fully concolution network(FCN) model. FCN model consists of mainly three parts: encoder, 1x1 convolution layer and decoder layer. 

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



## Experimentation: Testing in Simulation
1. Copy your saved model to the weights directory `data/weights`.
2. Launch the simulator, select "Spawn People", and then click the "Follow Me" button.
3. Run the realtime follower script
```
$ python follower.py my_amazing_model.h5
```

The output of my final testing in simulator is shown below:

[image_3]: ./docs/misc/follow.png
![alt text][image_3]
