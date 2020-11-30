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

The input image has a size of (64,64,3). 

For purpose of demonstration, the model is trained for 2 epoches firstly. The model requires several hours of training with the ResNet. So the model will use the pre-trained model instead which is saved as *ResNet50.h5* file. In this way, lots of time is saved. The result of the prediction is saved in the *Result.CSV* file.

You can try to put your image inside for prediction. To test your own sign image, only need to change the file name in line 239.


Keep safe and see you soon!

## Results

#### Performance on the Test set

**Test Accuracy**	0.866

**Test Loss** 0.530

**Rsult of prediction of my_image.jpg** :[3.4187701e-06,2.7741256e-04,9.9952292e-01,1.9884241e-07,1.9561907e-04,4.1168590e-07]

## Reference
1. [Deep Residual Networks](https://github.com/KaimingHe/deep-residual-networks#table-of-contents)
2. [Trained image classification models for Keras](https://github.com/fchollet/deep-learning-models)



