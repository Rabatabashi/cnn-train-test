#!/usr/bin/python3
#
# This python script train a fully connected neural network with multiple layers.
#	- Command line parameter is the directory path, which contains the mnist dataset.
#	- This script load the mnist images and labels into variables.
#	- It creates an instance of CNN and traing that with the parameters below and the previously mentioned images.
#
# author Nagy Marton

import argparse			#argumentum parser for give command line parameters.
import numpy as np		#numpy for mathematical operations and store the variables.
from mnist import MNIST		#this module handle the mnist data (load, display, etc.)
import cnnClass as cnn		#import module ./cnnClass.py. Our own cnn class.



#ARGPARSER
parser = argparse.ArgumentParser(description='Arguments of main.py which trains a cnn with MNIST data.')

parser.add_argument("--mnistDir", required=True,  type=str, help="The path of directory which includes the MNIST files.")
args = parser.parse_args()

mnistDir = args.mnistDir



mndata = MNIST(mnistDir)

#Load training images (for train)
trainImages, trainLabels = mndata.load_training()
#Load testing images (for evaluation)
testImages, testLabels = mndata.load_testing()

numberOfTrainExample = len(trainImages)
print(numberOfTrainExample)
numberOfTestExample = len(testImages)
print(numberOfTestExample)


##Parameters of CNN
#Initialization of the weights and biases with initializationType mezhod.
#At now only the "UniformRandom" is allowed.
initializationType = "UniformRandom"

#layerTypeList is a python list object which elements can be Convolution or FullyConnected.
#The convolutionl layer contains N (count) x 3D convolutional kernels (FH, FW, C) which works on their 3D input activation (H, W, C).
#Fully Connected layer contains a 2D (H, W) weight matrix (+biases (count like output)), where the W represents the flattened input activation map all pixels and H is the output vector elements.
layerTypeList = ["Convolution", "Convolution", "FullyConnected"]

#activationTypeList is a python list which have same size with layerTypeList.
#This list contains the activation function type for each layer. [Nonlinearities]
#These can be: "sigmoid", "linear", "relu", "softmax"
#NOTE: The "softmax" derivate function is not same with the literature. It is not reliable. Avoid to use it yet.
activationTypeList = ["sigmoid", "sigmoid", "sigmoid"]

#convolutional layer pattern [N, H, W, C], [count, height, width, channels] of kernels.
#fully connected layer pattern [K, C], [all pixels of previous layer, output count] of kernels.
#WARNING: The Ni and Ci must be same between two consecutive convolutional layers.
weightSizeList = [[10, 3, 3, 1], [10, 3, 3, 10], [10*7*7, 10]]


#It creates an instance of CNN from cnnClass.py
#Initialization of network
CNN = cnn.CNN(initializationType, layerTypeList, activationTypeList, weightSizeList)

#TRAIN
#Select images from all
#For the test of train, we did not use the whole 60000 images.
numberOfTrainExample = 200
xx = [None] * numberOfTrainExample
yy = [None] * numberOfTrainExample
for i in range(numberOfTrainExample):
	#xx is a list which contains selected images in 28x28 numpy array
	xx[i] = np.reshape(np.asarray(trainImages[i], dtype=np.float128), (28, 28, 1))
	yy[i] = np.asarray(trainLabels[i])

#Training of neural network with those lists, parameters images and labels.
CNN.train(xx, yy, batchSize=1, numberOfEpochs=5, lr=0.0001)