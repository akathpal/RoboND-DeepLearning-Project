[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)
[image_0]: ./docs/misc/train1.png
[image_1]: ./docs/misc/train2.png
[image_2]: ./docs/misc/train3.png
[image_3]: ./docs/misc/follow.png


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

![alt text][image_0] 

![alt text][image_1] 

![alt text][image_2] 

## Image Preprocessing

The next step after collecting the data is to preprocess the data which converts the depth masks into binary masks suitable for training the neural network. In our case, we dont require any depth information of how far the target is in the image. Our goal is to just detect if there are pixels similar to target in the image or not. 
Image preprocess function also converts png to jpeg format for reducing the size of the dataset.

## Model Arhitecture

The next step is to make a fully concolution network(FCN) model. FCN model consists of mainly three parts: encoder, 1x1 convolution layer and decoder layer. 

## Hyper Parameters



## Experimentation: Testing in Simulation
1. Copy your saved model to the weights directory `data/weights`.
2. Launch the simulator, select "Spawn People", and then click the "Follow Me" button.
3. Run the realtime follower script
```
$ python follower.py my_amazing_model.h5
```

The output of my final testing in simulator is shown below:

![alt text][image_3]
