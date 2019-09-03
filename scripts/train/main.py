#!/usr/bin/env python

# Loads the mnist images and labels and trains a fully connected neural network with multiple layers.
#
# @author Nagy Marton
# @author Kishazi "janohhank" Janos

# This module provides a portable way of using operating system dependent functionality.
import os
# Command lien argumentum parser library.
import argparse
# Scientific computing library.
import numpy
# Database provider library.
import mnist
# Our convulotional neural network abstraction.
import CNN

parser = argparse.ArgumentParser(description="TODO")
parser.add_argument("--mnistDir",required=True,type=str,help="The path of a directory which includes the MNIST files.")
args = parser.parse_args()

mnistDir = args.mnistDir

if(os.path.isdir(mnistDir) is False):
	raise Exception("The mnistDir input parameter does not denote a directory: " + mnistDir)

print("[" + __file__ + "]" + "[INFO]" + " Loading MNIST data.")
mnistData = mnist.MNIST(mnistDir)

# Load training images (for train)
trainImages, trainLabels = mnistData.load_training()
# Load testing images (for evaluation)
testImages, testLabels = mnistData.load_testing()

numberOfTrainExample = len(trainImages)
numberOfTestExample = len(testImages)
print("[" + __file__ + "]" + "[INFO]" + " Loaded train images count:",numberOfTrainExample)
print("[" + __file__ + "]" + "[INFO]" + " Loaded evaluation images count:",numberOfTestExample)

print("[" + __file__ + "]" + "[INFO]" + " Initializing CNN class.")
# Weights and biases initialization method type. At now only the "UniformRandom" is allowed.
initializationType = "xavierUniform"

# Describe the layers type and count in a list. The list element can be "Convolution" or "FullyConnected".
#	* A convolutional layer contains N (count) x 3D convolutional kernels (FH, FW, C) which works on their 3D input activation (H, W, C).
#	* A fully connected layer contains a 2D (H, W) weight matrix (+ biases - count like output),
#	  where the W represents the flattened input activation map all pixels and H is the output vector elements.
layerTypeList = ["Convolution", "Convolution", "FullyConnected"]

# Describe the activation function types for each layer in a list. The list element can be "sigmoid", "linear", "relu", "softmax".
# NOTE: The "softmax" derivate function is not same with the literature. It is not reliable. Avoid to use it yet.
activationTypeList = ["relu", "sigmoid", "softmax"]

# Describe convolutional layer patterns [N, H, W, C], [count, height, width, channels] of kernels.
# Describe fully connected layer pattern [K, C], [all pixels of previous layer, output count] of kernels.
# WARNING: The Ni and Ci must be same between two consecutive convolutional layers.
weightSizeList = [[20, 5, 5, 1], [20, 3, 3, 20], [20*7*7, 10]]

# Initialization of a network.
cnn = CNN.CNN(initializationType, layerTypeList, activationTypeList, weightSizeList)
print("[" + __file__ + "]" + "[INFO]" + " CNN initialized.")

# TODO image preprocessing need to be modified and moved into a class/utility script.
print("[" + __file__ + "]" + "[INFO]" + " Preprocessing images.")
numberOfTrainExample = 200
xx = [None] * numberOfTrainExample
yy = [None] * numberOfTrainExample
for i in range(numberOfTrainExample):
	# xx is a list which contains selected images in 28x28 numpy array
	xx[i] = numpy.reshape(numpy.asarray(trainImages[i], dtype=numpy.float128), (28, 28, 1))
	yy[i] = numpy.asarray(trainLabels[i])

print("[" + __file__ + "]" + "[INFO]" + " Start training.")
cnn.train(xx, yy, batchSize=1, numberOfEpochs=5, learningRate=0.0001)
print("[" + __file__ + "]" + "[INFO]" + " Training done.")