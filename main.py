#!/usr/bin/python3
#
# This python script train a fully connected neural network with multiple layers.
#
# author Nagy Marton

import argparse
import numpy as np


#ARGPARSER
#parser = argparse.ArgumentParser(description='Process some integers.')

#parser.add_argument("--inputParametersFile", required=True,  type=int, help="The path of file which includes the parameters of CNN and hyper-parameters of train.")
#parser.add_argument("--inputSize", required=True,  type=int, help="The dimension of input images. This will be the height and width of input of CNN.")  #NOTE: 28px x 28px at MNIST
#parser.add_argument("--countOfClasses", required=True,  type=int, help="Number of classes.")  #NOTE: 10 class at MNIST
#args = parser.parse_args()

#inputSize = args.inputSize
#countOfClasses = args.countOfClasses


#weightSizes = [
    #[5, 3, 3, 1],
    #[10, 3, 3, 5],
    #[(inputSize/(2**2))*(inputSize/(2**2)) * 10, countOfClasses]
#]

#print(weightSizes)

#HELP for maxpool backward
a = np.array([[1,2,3], [1, 455, 6], [1,11,1]])
b = (a == np.max(a))
c = np.array([[11,21,31], [11,41,61], [11,11,11]])
print(a)
print(b)
d = b*c
print(d)


#Help of softmax derivate
print()
print()
def softmax(inp):
	s = np.exp(inp) / np.sum(np.exp(x))
	return s

def softmax_grad(softmax):
	# Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
	s = softmax.reshape(-1,1)
	return np.diagflat(s) - np.dot(s, s.T)

x = np.array([1,2,3,4])
sm = softmax(x)
diagSM = np.diag(sm)
gradSM = softmax_grad(sm)
print(x)
print(sm)
print(diagSM)
print(len(diagSM))

print(gradSM)


