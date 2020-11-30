# Image-classification-models-with-Deep-Residual-Networks
Image classification models based on Keras, tensorflow backend

By Tianyang Zhao

## Table of Contents
1. Introduction
2. Models
3. Results
4. Reference

## Introduction
This repository contains the Deep Residual Network model. The goal of the model is to decipher sign language (from 0 -> 5) .
Here are examples for each number, and how an explanation of representing the labels. These are the original pictures, before lowering the image resolutoion to 64 by 64 pixels. 

![image](https://github.com/berlintofind/Multilayer-Perceptron/blob/main/images/hands.png)

#### Note 
Note that this is a subset of the SIGNS dataset. The complete dataset contains many more signs.

#### Packages Version
1. Python 3.6.9 
2. tensorflow-gpu 1.13.1
3. numpy 1.19.1
4. CUDA Version 10.1
5. scipy 1.2.1
6. keras-gpu 2.3.1


## Models
Deep "plain" networks don't work in practice because they are hard to train due to gradients exploration. ResNet block with skip connections can help to address this problem thus to make the networks go deeper. Two types of blocks are implemented in this model: the identity block and the convolutional block. Deep Residual Networks are built by stacking these blocks together.

The model contains 50 hidden layers. Using a softmax output layer, the models is able to generalizes more than two classes of outputs.

The architecture of the model is shown as follows:
![image](https://github.com/berlintofind/Image-classification-models-with-Deep-Residual-Networks/blob/main/images/resnet_kiank.png)

The inptu images has a size of (64,64,3). 


You can try to put your image inside, and test your own sign image by changing the file name in line 239.

#### Note
1. GPU memory might be insufficient for extremely deep models ( it takes 1GB, around 27min, trained on Tesla K80)
2. Changes of mini-batch size should impact accuracy ( minibatch_size = 32 in this model)
3. the data is randomly shuffled at the beginning of every epoch.

Keep safe and see you soon!

## Results
#### Performance on the Training set
**Train Accuracy** 0.999074

#### Performance on the Test set
**Test Accuracy**	0.866

**Test Loss** 0.530

## Reference
1. [Deep Residual Networks](https://github.com/KaimingHe/deep-residual-networks#table-of-contents)
2. [Trained image classification models for Keras](https://github.com/fchollet/deep-learning-models)



