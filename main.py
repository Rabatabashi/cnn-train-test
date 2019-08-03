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

##HELP for maxpool backward
#a = np.array([[1,2,3], [1, 455, 6], [1,11,1]])
#b = (a == np.max(a))
#c = np.array([[11,21,31], [11,41,61], [11,11,11]])
#print(a)
#print(b)
#d = b*c
#print(d)


##Help of softmax derivate
#print()
#print()
#def softmax(inp):
	#s = np.exp(inp) / np.sum(np.exp(x))
	#return s

#def softmax_grad(softmax):
	## Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
	#s = softmax.reshape(-1,1)
	#return np.diagflat(s) - np.dot(s, s.T)

#x = np.array([0.0,0.2,0.3,0.5])
#sm = softmax(x)
#diagSM = np.diag(sm)
#gradSM = softmax_grad(sm)
#print(x)
#print(sm)
#print(diagSM)
#print(len(diagSM))

#print(gradSM)
#print(np.sum(gradSM, 1))

##HELP for indexing
#print("indexing")
#print("x: ", x)
#print("x[-1]: ", x[-1])
#print("x[2:]: ", x[2:])
#print("x[0:2]: ", x[0:2])
#xL = 2
#xR = 4
#yB = 1
#yT = 3
#x = np.array([[1,2,3,4], [1,2,3,4]])
#xH, xW  = x.shape
#print(xH, xW)
#y = np.zeros((xH+yT+yB, xW+xL+xR))
#print(x)
#print(y)
#y[yT:-yB, xL:-xR] = x
#print(y)

##HELP multiple variable declaration
#xx = yy = zz = 3
#print(xx)
#print(yy)
#print(zz)

##HELP matrix multiplication and subtraction element-wise
#xxx = np.array([[1,2,1],[2,2,2],[3,4,6]])
#yyy = np.array([[8,2,4],[2,2,3],[3,1,2]])
#zzz = np.multiply(xxx,yyy)
#print(xxx)
#print(yyy)
#print(zzz)
#sss = np.subtract(xxx,(0.1*yyy))
#print(sss)
#a = np.array([1,2,3])
#xxx = np.array([[1,2,1],[2,2,2],[3,4,6]])
#print("SHAPE", a.shape)
#b = np.random.randn(a.shape[0])
#print(a)
#print(b)

#yyy = np.array([[8,2,4],[2,2,3],[3,1,2]])
#zz = xxx * yyy
#print(zz)

#HELP reverese range
#numberOfLAyers = 10
#for layerID in reversed(range(numberOfLAyers - 1)):
	#print(layerID)

##HELP for flatten and reshape	
#xxx = np.array([[1,2,1],[2,2,2],[3,4,6]])
#(h,w) = xxx.shape
#xShape = xxx.shape
#print(xShape)
#print(xxx)
#xf = xxx.flatten()
#print(xf)
#xrs = xf.reshape(h,w)
#print(xrs)

#HELP list dictionary combo
#lista = []
#for i in range(3):
	#dictionarya = {}
	#if i == 1:
		#dictionarya["a"] = 1
		#dictionarya["b"] = 2
		#dictionarya["c"] = 3
	#else:
		#dictionarya["a"] = 1
		#dictionarya["b"] = 2
	#lista.append(dictionarya)
#print(lista)