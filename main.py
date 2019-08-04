#!/usr/bin/python3
#
# This python script train a fully connected neural network with multiple layers.
#
# author Nagy Marton

import argparse
import numpy as np
from mnist import MNIST		#this module handle the mnist data (load, display, etc.)
import cnnClass as cnn		#import module ./cnnClass.py. Our own cnn class.



#ARGPARSER
parser = argparse.ArgumentParser(description='Arguments of main.py which trains a cnn with MNIST data.')

parser.add_argument("--mnistDir", required=True,  type=str, help="The path of directory which includes the MNIST files.")
args = parser.parse_args()

mnistDir = args.mnistDir


#
mndata = MNIST(mnistDir)

#Load training images (for train)
trainImages, trainLabels = mndata.load_training()
#Load testing images (for evaluation)
testImages, testLabels = mndata.load_testing()


numberOfTrainExample = len(trainImages)
print(numberOfTrainExample)
numberOfTestExample = len(testImages)
print(numberOfTestExample)

initializationType = "Xavier"
layerTypeList = ["Convolution", "Convolution", "FullyConnected"]
activationTypeList = ["sigmoid", "sigmoid", "sigmoid"]
weightSizeList = [[10, 3, 3, 1], [10, 3, 3, 10], [10*7*7, 10]]
#weightSizeList = [[10, 3, 3, 1], [10, 3, 3, 10], [10*7*7, 10]]

CNN = cnn.CNN(initializationType, layerTypeList, activationTypeList, weightSizeList)

#TRAIN
#Select images from all
#xx is alist which contains selected images in 28x28 numpy array
numberOfTrainExample = 200
xx = [None] * numberOfTrainExample
yy = [None] * numberOfTrainExample
for i in range(numberOfTrainExample):
	xx[i] = np.reshape(np.asarray(trainImages[i], dtype=np.float128), (28, 28, 1))
	yy[i] = np.asarray(trainLabels[i])

CNN.train(xx, yy, batchSize=1, numberOfEpochs=5, lr=0.0001)

#index = np.random.randint(numberOfTrainExample)  # choose an index ;-)
#print(mndata.display(trainImages[index]))
#print()
#print(trainLabels[index])

#shapeOfImage = np.asarray(trainImages[index]).shape
#print(shapeOfImage)


###numberOfLayers = len(layerTypeList)
###inputH = inputW = 28
###fmH = fmW = inputH / (np.pow(2, numberOfLayers-1))