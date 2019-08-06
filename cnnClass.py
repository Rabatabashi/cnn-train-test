#!/usr/bin/python3
#
# This python scripts contains two class and their functions.
#	1. class is the layer.
#	2. class is the convolutional neural network.
# Lots of usefull functions are implemented in this script.
# These functions do the forward, backward, weights update and those operations which are necessary for these.
# The CNN.train() function (at the end of code) is the main function of this code. With that and some hyperparameters we can train this CNN.
#
# author Nagy Marton

import numpy as np


#This is the Layer class which called by the CNN class.
#The member variables of it contains information, which describe a CNN layer.
#Furthermore the weights and biases, which are trainable parameters of CNN, initialize and stored in this class.
class Layer:
	def __init__(self, layrID, initializationType, layerType, activationType, weightSize):
		self.layrID = layrID					#It means the sequence number of layer. Later it will use for identify.
		self.initializationType = initializationType		#only UniformRnd allowed yet
		self.layerType = layerType				#convolution or fully connected
		self.activationType = activationType			#nonlinearity of layer
		self.weights = {}					#trainable parameters
		self.biases = {}					#trainable parameters
		if self.initializationType == "UniformRandom":
			if self.layerType == "Convolution":
				#[N,H,W,C] is the convention.
				numberOfKernels = weightSize[0]
				kernelHeight = weightSize[1]
				kernelWidth = weightSize[2]
				channels = weightSize[3]
				for kernelIdx in range(numberOfKernels):
					self.weights[kernelIdx] = np.random.uniform(-0.01, +0.01, (kernelHeight, kernelWidth, channels))
					self.weights[kernelIdx]= self.weights[kernelIdx].astype(dtype=np.float128)
					self.biases[kernelIdx] = np.random.uniform(-0.01, +0.01, (1))
					self.biases[kernelIdx]= self.biases[kernelIdx].astype(dtype=np.float128)
			elif self.layerType == "FullyConnected":
				lastFeatureMapSize = weightSize[0]	#TODO it must calculate instead of arbitrary give it.
				numberOfOutPut = weightSize[1]
				self.weights[0] = np.random.uniform(-0.01, +0.01, (lastFeatureMapSize, numberOfOutPut))
				self.weights[0]= self.weights[0].astype(dtype=np.float128)
				self.biases[0] = np.random.uniform(-0.01, +0.01, (numberOfOutPut))
				self.biases[0]= self.biases[0].astype(dtype=np.float128)
		else:
			print("initializationType =", self.initializationType)
			raise Exception("This initializationType is not defined yet. Only UniformRandom is allowed.")

	def __del__(self): 
		print('Destructor called, Layer deleted.')

#This is the class of CNN (convolutional neural network).
#The member variable of this class is the layers list.
#The elements of this list is instance of the Layer class.
#For example if we want to search the 20th kenrel at the 2nd layer, which is a convolution layer, 
# we call self.layers[1].weights[19], which is a 3D numpy.array.
class CNN:
	layers = []
	def __init__(self, initializationType, layerTypeList, activationTypeList, weightSizeList):
		numberOfLayers = len(layerTypeList)	#all lists in the arguments of this function have same number of elements.
		for layer in range(numberOfLayers):
			self.layers.append(Layer(layer, initializationType, layerTypeList[layer], activationTypeList[layer], weightSizeList[layer]))

	
	def __del__(self): 
		print('Destructor called, CNN deleted.')


########################################################################################################################
##Padding
#Padding means fill zero elements.
	#If we want to get same output sizes after the convolution, like it were before, ...
	#... this function determines the those parameters of padding. (xLeft, xRight, yBot, yTop)
	#This is a naive function only works on the symmetric input and symmetric kernels.
	def doPaddingParameterDetermination(self, inputFeatureMap, layerID):
		#The base equation of the size of next feature map is: O = ((I - K + 2P) / S) + 1
		#,where O: output size; I: input size; K: kernel size; P: pading size; S: kernel stride
		#If we want to same size we need to I = O and we sort the equation to P [P = (I*S - S - I + K) / 2]
		(fH,fW,fC) = self.layers[layerID].weights[0].shape
		#NOTE: We handle only symmetric kernels now, later it will be different aspect ratio too.
		kernelSizes = fH
		
		#NOTE: WE use one kernel strides, later it will be a parameters too.
		kernelStride = 1
		
		shapeOfFeatureMap = inputFeatureMap.shape
		
		P = int(np.ceil(((shapeOfFeatureMap[0] * kernelStride) - kernelStride - shapeOfFeatureMap[0] + kernelSizes) / 2))

		#We return P in thise case it is equal with all parameters (xLeft, xRight, yBot, yTop) of padding. 
		return P


	#This function create a padding around the feature map.
	def doPaddingOnLayer(self, inputFeatureMap, layerID):
		#Call the naive parameter determinator function.
		padding = self.doPaddingParameterDetermination(inputFeatureMap, layerID)
					    
		yBot = yTop = xLeft = xRight = padding
		
		#inputFeatureMap is a 3D tensor. (input feature map.
		#Handle different to the opposite sides of the tensor, because the kernel sizes can be even and if the feature map is not even it can be indexing out frominput.
		#assymetric kernel sizes make it neccessary of the neighboor  sides should be different have different shapes at paadding.
		shapeOfFeatureMap = inputFeatureMap.shape
		H = shapeOfFeatureMap[0]
		W = shapeOfFeatureMap[1]
		C = shapeOfFeatureMap[2]
		
		newH = H + yBot + yTop
		newW = W + xLeft + xRight
		outputFeatureMap = np.zeros((newH, newW, C), dtype=np.float128)
		outputFeatureMap[yTop:-yBot, xLeft:-xRight, :] = inputFeatureMap	#WARNING: these function does not handle the batchsize yet.
		
		cache = padding
		
		return outputFeatureMap, cache
	

########################################################################################################################
##Conv and FC
	#Convolutional layer fowrard.
	#This function convolutions for all kernels at a layer.
	#The output of this function is a feature map.
	#It fills a cache variable with inputs and weigths, because at the backpropagation it will use for further error determination.
	def doConvolutionsOnLayer(self, inputFeatureMap, layerID):
		#Filter/Kernel sizes
		(fH,fW,fC) = self.layers[layerID].weights[0].shape
		
		#Feature map sizes
		outputDepth = len(self.layers[layerID].weights)
		(inputWidth, inputHeight, inputChannels) = inputFeatureMap.shape
		outputHeight = inputHeight - fH + 1
		outputWidth = inputWidth - fW + 1
		
		outputFeatureMap = np.zeros((outputHeight, outputWidth, outputDepth), dtype=np.float128)
		
		for kernelIdx in range(outputDepth):
			for h in range(outputHeight):
				for w in range(outputWidth):
					xSlice = inputFeatureMap[h:h+fH, w:w+fW, :]
					outputFeatureMap[h, w, kernelIdx] = np.sum(xSlice * self.layers[layerID].weights[kernelIdx])
		
		#for backpropagation we need the X and W tensors
		cache = (inputFeatureMap, self.layers[layerID].weights.copy())
		
		return outputFeatureMap, cache

	#Backward convoltuion
	#During backpropagation we need to propagate the errors of weights, biases and the inputActivation.
	#This function determine these (dW, dB, dX) from delta, which comes from the next layer dX and the activation function of the current layer.
	#The dX is just a numpy.array, but the dW and dB are stored in a list, which will be store (later in the code) another list.
	def backwardConvolution(self, layerID, delta, cache):
		
		#Stored feature map
		X = cache["X"]
		
		#Stored wights
		W = cache["Weights"]

		shapeOfAWeightKernel = W[0].shape
		fH = shapeOfAWeightKernel[0]
		fW = shapeOfAWeightKernel[1]
		#fC = shapeOfAWeightKernel[2]

		#NOTE: NN = N and CC = C, because pool decrease only the width and height of feature map
		#NN, HH, WW, CC = delta.shape()
		deltaShape = delta.shape
		Hcurr = deltaShape[0]
		Wcurr = deltaShape[1]
		CC = deltaShape[2]
		
		#These list will be store the derivates of the weights and biases.
		dWList = []
		dBList = []
		
		dX = np.zeros(X.shape, dtype=np.float128)
		
		#Feature map sizes
		countOfKernels = len(self.layers[layerID].weights)
		for kernelIdx in range(countOfKernels):
			#dE/dW
			dW = np.zeros_like(self.layers[layerID].weights[kernelIdx], dtype=np.float128)	#NOTE: If all kernels are same this line can goto out from this for cycle.
			#dE/dB
			dB = np.zeros_like(self.layers[layerID].biases[kernelIdx], dtype=np.float128)	#NOTE: If all kernels are same this line can goto out from this for cycle.
			for h in range(Hcurr):
				for w in range(Wcurr):
					dX[h:h+fH, w:w+fW, :] += W[kernelIdx] * delta[h,w,kernelIdx]	#Only thos elementsof dX accumulates, which contributed to delta[h,w,kernelIdx] elements.
					dW += X[h:h+fH, w:w+fW, :] * delta[h,w,kernelIdx]
					dB += delta[h,w,kernelIdx]
			dWList.append(dW)
			dBList.append(dB)
		
		
		return dX, dWList, dBList

	#Fully Connected layer forward.
	#This function multiply the input activation(1D) with the weights of current layer(2D) and add to that the biases(1D).
	def doFullyConnectedOperationOnLayer(self, inputFeatureMap, layerID):
		#Create 1D vector from 3D feature map
		#.flatten() do it if inputFeatureMap is a numpy.array
		X = inputFeatureMap.copy()
		xShape = X.shape
		#X.flatten()
		X = np.reshape(X, (xShape[0] * xShape[1] * xShape[2]))

		(countOfActivation, countOfClasses) = self.layers[layerID].weights[0].shape

		outputVector = np.add(np.matmul(X, self.layers[layerID].weights[0]), self.layers[layerID].biases[0])
		
		#for backpropagation we need the X and W tensors
		cache = (X, xShape, self.layers[layerID].weights.copy(), self.layers[layerID].biases.copy())
		
		return outputVector, cache

	#Backward Fully Conected layer
	#Determine the dX, dW, dB (mentioned above at the backwardConvolution()). The store of outputs 
	def backwardFullyConnected(self, layerID, delta, cache):
		
		#Stored feature map (it is an 1D)
		X = cache["X"]
		#Stored wights
		W = cache["Weights"]
		
		#These list will be store the derivates of the weights and biases.
		dWList = []
		dBList = []
		
		dW = np.outer(X, delta)
		dW= dW.astype(dtype=np.float128)
		dB = delta
		dB= dB.astype(dtype=np.float128)
		
		dX = np.dot(W[0], delta)
		dX= dX.astype(dtype=np.float128)
		
		dWList.append(dW)
		dBList.append(dB)
		
		return dX, dWList, dBList


########################################################################################################################
##Bias
	#Add biases to all elements of the input
	#The inputs of this function is the putput of the convolution layer so the output of this function is the logits.
	#The logits is the input of activation function (commom notation is z).
	def doBiasAddOnLayer(self, inputFeatureMap, layerID):
		#Feature map sizes
		outputFeatureMap = np.zeros_like(inputFeatureMap, dtype=np.float128)
		outputDepth = len(self.layers[layerID].biases)
		for kernelIdx in range(outputDepth):
			outputFeatureMap[:,:,kernelIdx] = inputFeatureMap[:,:,kernelIdx] + self.layers[layerID].biases[kernelIdx]
		
		#for backpropagation we need the X and W tensors
		cache = self.layers[layerID].biases.copy()
		
		return outputFeatureMap, cache

########################################################################################################################
##Activations(/Nonlinearities) and their derivates (derivates are neccessary during backpropagation).
	
	#This function contains same activation functions then getActivationFunction()[below], but this function is not use lambdas.
	def getActivationFunctionWithoutLambda(self, name, X):
		if(name == "sigmoid"):
			y = np.copy(X)
			results = np.exp(y)/(1+np.exp(y))
			return results
		elif(name == "linear"):
			return X
		elif(name == "relu"):
			y = np.copy(X)
			y[y<0] = 0
			return y
		elif(name == "softmax"):
			xx = np.copy(x)
			y = np.exp(xx) / np.sum(np.exp(xx))
			return y
		else:
			print('Unknown activation function. linear is used')
			return lambda x: x
		return

	
	#This function return the selected activation function.
	#It is called by doActivationFunctionOnLayer() function.
	def getActivationFunction(self, name):
		if(name == "sigmoid"):
			return lambda x : np.exp(x)/(1+np.exp(x))
		elif(name == "linear"):
			return lambda x : x
		elif(name == "relu"):
			def relu(x):
				y = np.copy(x)
				y[y<0] = 0
				return y
			return relu
		elif(name == "softmax"):
			def softmax(x):
				xx = np.copy(x)
				y = np.exp(xx) / np.sum(np.exp(xx))
				return y
			return softmax
				
		else:
			print('Unknown activation function. linear is used')
			return lambda x: x

	#Nonlinearity or Activation function
	#It is waiting the results of conv + bias (=logits) and return with the activation of the selected nonlinearity type.
	def doActivationFunctionOnLayer(self, inputFeatureMap, typeOfNonlinearity):
		
		#Select nonlinearity function
		activationFunction = self.getActivationFunction(typeOfNonlinearity)
		
		#Use activation function on input feature map
		outputFeatureMap = activationFunction(inputFeatureMap)
		
		##outputFeatureMap = self.getActivationFunctionWithoutLambda(typeOfNonlinearity, inputFeatureMap)
		
		return outputFeatureMap
    
    
	#At the backpropagation we should use these functions instead of the original activation functions.
	#These are the derivatives of the original ones.
	def getDerivitiveActivationFunction(self, name):
		if(name == "sigmoid"):
			sig = lambda x : np.exp(x)/(1+np.exp(x))
			return lambda x :sig(x)*(1-sig(x))
		elif(name == "linear"):
			return lambda x: 1
		elif(name == "relu"):
			def relu_diff(x):
				y = np.copy(x)
				y[y>=0] = 1
				y[y<0] = 0
				return y
			return relu_diff
		elif(name == "softmax"):
			softmax = self.getActivationFunction('softmax')
			return lambda x :softmax(x)*(1-softmax(x))		#NOTE: the description was different from it.
		else:
			print('Unknown activation function. linear is used')
			return lambda x: 1

	#Select derivates of Nonlinearity (or Activation) function
	#It is waiting the results of conv + bias (=logits) and return with the activation of the selected derivates of the nonlinearity.
	def doDerivateOfActivationFunctionOnLayer(self, inputFeatureMap, typeOfNonlinearity):
		
		#Select the derivate funciton of current nonlinearity.
		# d sigma(x) / dx, whgere sigma(x) is the activation function.
		derivateOfActivationFunction = self.getDerivitiveActivationFunction(typeOfNonlinearity)
		
		#Use activation function on input feature map
		outputFeatureMap = derivateOfActivationFunction(inputFeatureMap)

		return outputFeatureMap
	

########################################################################################################################
#Poolings
	#Max-pooling layer - reduce the height and width of feature map, (select the maximal activity in the kernel slices)
	def doMaxPoolingOnlayer(self, inputFeatureMap, kernelHeight, kernelWidth, strideY, strideX):
		(inputWidth, inputHeight, inputChannels) = inputFeatureMap.shape
		#determine the output resolution after pooling
		outputHeight = int(np.ceil((inputHeight - kernelHeight) / strideY) + 1)
		outputWidth = int(np.ceil((inputWidth - kernelWidth) / strideX) + 1)
		
		outputFeatureMap = np.zeros((outputHeight, outputWidth, inputChannels), dtype=np.float128)
		
		for c in range(inputChannels):
			for h in range(outputHeight):
				for w in range(outputWidth):
					#TODO indexing out from input we should handle those foreign pixels
					xSlice = inputFeatureMap[(h*strideY):(h*strideY)+kernelHeight, (w*strideX):(w*strideX)+kernelWidth, c]
					outputFeatureMap[h, w, c] = np.max(xSlice)
		
		parametersOfPooling = {}
		parametersOfPooling["stride"] = strideY	#NOTE: it is same than strideX
		parametersOfPooling["poolingFilterHeight"] = kernelHeight
		parametersOfPooling["poolingFilterWidth"] = kernelWidth
		#for backpropagation
		cache = parametersOfPooling
		
		return outputFeatureMap, cache

	#It is a naive implementation of backward pooling based on:
	#https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/pooling_layer.html
	def doMaxPoolingOnlayerBackWard(self, pooledDeltaX, cache, xShape=None):
		
		#shape of the output feature map. This is the backpropagated error tensor.
		#It contain the error of next feature map and the derivates of the nonlinearity(wrt the logits)
		if not (xShape is None):
			#if the delta come from a fully connected it is a 1D vector
			#Reshape it into 3D tensor
			pooledDeltaX = np.reshape(pooledDeltaX, (xShape))
			#(Hcurr, Wcurr, Ccurr) = delta.shape
			#NOTE: NN = N and CC = C, because pool decrease only the width and height of feature map
			#NN, HH, WW, CC = pooledDeltaX.shape()
			pooledDeltaXShape = pooledDeltaX.shape
			HH = pooledDeltaXShape[0]
			WW = pooledDeltaXShape[1]
			CC = pooledDeltaXShape[2]
		else:
			#The delta come from convoluton
			#NOTE: NN = N and CC = C, because pool decrease only the width and height of feature map
			#NN, HH, WW, CC = pooledDeltaX.shape()
			pooledDeltaXShape = pooledDeltaX.shape
			HH = pooledDeltaXShape[0]
			WW = pooledDeltaXShape[1]
			CC = pooledDeltaXShape[2]
		
		
		unpooledX = cache["Activations"]
		parametersOfPooling = cache["PoolParameters"]
		
		#N is the batchSize.
		#H, W, C is the height width channels of feature map which was pooled.
		#N, H, W, C = unpooledX.shape()
		unpooledShape = unpooledX.shape
		H = unpooledShape[0]
		W = unpooledShape[1]
		C = unpooledShape[2]
		
		stride = parametersOfPooling["stride"]
		fHeight = parametersOfPooling["poolingFilterHeight"]
		fWidth = parametersOfPooling["poolingFilterWidth"]

		unPooledDeltaX = np.zeros_like(unpooledX, dtype=np.float128)
		
		#for n in range(N):
		for c in range(C):
			for y in range(HH):
				for x in range(WW):
					xSlice = unpooledX[y*stride:y*stride+fHeight, x*stride:x*stride+fWidth, c]
					xSlice = np.squeeze(xSlice)
					#Select thos i,j which is (/are) the maximal values of input, we propagates only those errors.
					selectMax = np.where(xSlice >= np.amax(xSlice), 1, 0)
					
					unPooledDeltaX[y*stride:y*stride+fHeight, x*stride:x*stride+fWidth, c] = selectMax * pooledDeltaX[y,x,c]
		return unPooledDeltaX

########################################################################################################################
##Loss function
	#This loss function compute the absolute value of loss
	def lossFunction(self, output, target):
		#target is a scalaer betweeen 0 and 9
		#we transfrom this scalar into a binary vector with 10 elements.
		targetVector = np.zeros(len(output), dtype=np.float128)
		targetVector[target-1] = 1.0
		print(output)
		print(targetVector)

		loss = np.sum(np.absolute(np.subtract(output, targetVector)))
		return loss

########################################################################################################################
##Gradients of Weights and Biases Summarizer
	#This function sume up dWs and dBs for images (in a batch)
	def doWeightsAndBiasSum(self, dWAll, dBAll, dWImg ,dBImg):
		numberOfLayers = len(self.layers)
		for layerID in range(numberOfLayers):
			kernelCounts = len(self.layers[layerID].weights)
			for kernelIdx in range(kernelCounts):
				dWAll[layerID][kernelIdx] = np.add(dWAll[layerID][kernelIdx], dWImg[layerID][kernelIdx])
				dBAll[layerID][kernelIdx] = np.add(dBAll[layerID][kernelIdx], dBImg[layerID][kernelIdx])
		return dWAll, dBAll


########################################################################################################################
#Forward and Backward
	#This function get the input and propagate the information forward to the output of CNN.
	#It is a pipeline. The output of each function is the input of the next one.
	#A for cycle iterate over all layers.
	#Create a list which cache all usefull results during forward run of CNN.
	#For example if we want to find the 6th layer weights.lists, it seems like: cacheList[5]["Weights"]
	#that is a list which each elements are a 3D numpy array.
	def forward(self, X):
		predictions = [None] * len(X)
		#TODO: This for cycle is necessary because we did not handle the batch list yet
		#The cacheList is not enough in this form if we use batch load in.
		for img in range(len(X)):
			cacheList = []
			#first input
			#it called output because later we will call the functions with these name.
			output = X[img]
			#For cycle to create the cnn pipeline.
			numberOfLAyers = len(self.layers)
			for layerID in range(numberOfLAyers):
				if self.layers[layerID].layerType == "Convolution":
					cacheMap = {}
					#print("BEFORE Padding:", output)
					output, cache = self.doPaddingOnLayer(output, layerID)
					#print("AFTER Padding:", output)
					cacheMap["Padding"] = cache
					output, cache = self.doConvolutionsOnLayer(output, layerID)
					#print("AFTER CONV:", output)
					cacheMap["X"] = cache[0]
					cacheMap["Weights"] = cache[1]
					#if layerID == 1:
						#print(cacheMap["Weights"][1])
					output, cache = self.doBiasAddOnLayer(output, layerID)
					cacheMap["Biases"] = cache
					cacheMap["Z"] = output							#It is the logits before activation function.
					#print("AFTER CONV+bias:", output)
					output = self.doActivationFunctionOnLayer(output, self.layers[layerID].activationType)
					cacheMap["Activations"] = output
					#print("AFTER SIGMOID:", output)
					output, cache = self.doMaxPoolingOnlayer(output, 2, 2, 2, 2)
					#print("AFTER MaxPool:", output)
					cacheMap["PoolParameters"] = cache
					cacheList.append(cacheMap)
				elif self.layers[layerID].layerType == "FullyConnected":			#WARNING: fully connected before a convolutional layer is not working, check this.
					cacheMap = {}
					output, cache = self.doFullyConnectedOperationOnLayer(output, layerID)
					#print("after FC:", output)
					cacheMap["X"] = cache[0]
					cacheMap["Shape"] = cache[1]
					cacheMap["Weights"] = cache[2]
					cacheMap["Biases"] = cache[3]
					cacheMap["Z"] = output
					cacheList.append(cacheMap)
					output = self.doActivationFunctionOnLayer(output, self.layers[layerID].activationType)
					#print("AFTER SM:", output)
			predictions[img] = output
		return predictions, cacheList




	#backpropagation
	#This is a pipeline similar to forward run.
	#In this case we propagate the errors from at the end of cnn to the inputs.
	#The propagation happened with a method which called chain rule. It is a chan if derivates.
	def backpropagation(self, predictions, target, cacheList):
		numberOfLayers = len(self.layers)
		dWAll = [None] * numberOfLayers
		dBAll = [None] * numberOfLayers
		
		totalLoss = 0
		#TODO: This for cycle is necessary because we did not handle the batch list yet
		for img in range(len(target)):
			currTarget = target[img]
			currentPrediction = predictions[img]
			#print(currentPrediction)
			#print(currTarget)
			loss = self.lossFunction(currentPrediction, currTarget)
			
			print(loss)
			
			#deltas = dE/dZ  it is the error for each layer
			deltas = [None] * numberOfLayers
			deltas[-1] = (loss)*(self.doDerivateOfActivationFunctionOnLayer((cacheList[-1]["Z"]), self.layers[-1].activationType))

			dWImg = [None] * numberOfLayers
			dBImg = [None] * numberOfLayers

			for layerID in reversed(range(numberOfLayers)):
				if self.layers[layerID].layerType == "Convolution":
					#Check that the following layer after convolution is a fully connected
					if self.layers[layerID + 1].layerType == "FullyConnected":
						xShape = cacheList[layerID + 1]["Shape"]
						dXLayer = self.doMaxPoolingOnlayerBackWard(deltas[layerID+1], cacheList[layerID], xShape)
						dXLayer, dWLayer, dBLayer = self.backwardConvolution(layerID, dXLayer, cacheList[layerID])
						padding = cacheList[layerID]["Padding"]
						dXLayer = dXLayer[padding:-padding, padding:-padding]	#decrease the size of feature map with paddings
					else:
						dXLayer = self.doMaxPoolingOnlayerBackWard(deltas[layerID+1], cacheList[layerID])
						dXLayer, dWLayer, dBLayer = self.backwardConvolution(layerID, dXLayer, cacheList[layerID])
						padding = cacheList[layerID]["Padding"]
						dXLayer = dXLayer[padding:-padding, padding:-padding]	#decrease the size of feature map with paddings
					#dX and the derivates of the nonlinearity have same shapes.
					deltas[layerID] = dXLayer * (self.doDerivateOfActivationFunctionOnLayer((cacheList[layerID]["Z"]), self.layers[layerID].activationType))
					dWImg[layerID] = dWLayer
					dBImg[layerID] = dBLayer
				elif self.layers[layerID].layerType == "FullyConnected":
					if layerID == (numberOfLayers - 1):
						startDelta = (loss)*(self.doDerivateOfActivationFunctionOnLayer((cacheList[layerID]["Z"]), self.layers[layerID].activationType))
						dXLayer, dWLayer, dBLayer = self.backwardFullyConnected(layerID, startDelta, cacheList[layerID])
						#dX and the derivates of the nonlinearity have same shapes.
						deltas[layerID] = dXLayer
					else:
						dXLayer, dWLayer, dBLayer = self.backwardFullyConnected(layerID, deltas[layerID+1], cacheList[layerID])
						#dX and the derivates of the nonlinearity have same shapes.
						deltas[layerID] = dXLayer * (self.doDerivateOfActivationFunctionOnLayer((cacheList[layerID]["Z"]), self.layers[layerID].activationType))
					dWImg[layerID] = dWLayer
					dBImg[layerID] = dBLayer
			totalLoss += loss
			if img == 0:
				dWAll = dWImg
				dBAll = dBImg
			else:
				dWAll, dBAll = self.doWeightsAndBiasSum(dWAll, dBAll, dWImg ,dBImg)
			
		return dWAll, dBAll, totalLoss

########################################################################################################################
##Weights and Biases UPDATES
	#This function updates the weights and biases depending on learning rate and determined dW and dB.
	def doUpdateWeightsAndBiases(self, dW, dB, lr):
		numberOfLayers = len(self.layers)
		for layerID in range(numberOfLayers):
			kernelCounts = len(self.layers[layerID].weights)
			for kernelIdx in range(kernelCounts):
				self.layers[layerID].weights[kernelIdx] = np.subtract(self.layers[layerID].weights[kernelIdx], (lr * dW[layerID][kernelIdx]))
				self.layers[layerID].biases[kernelIdx] = np.subtract(self.layers[layerID].biases[kernelIdx], (lr * dB[layerID][kernelIdx]))


########################################################################################################################
##TRAIN
	#This is the train function which will be called at the main.
	#This function call all functions which call the lower level ones.
	#train() function load the input image into a batch which contains only 1 image per batch now (the higher batchsize is not working yet.).
	#the loaded batch go through the cnn and the caches and outputs generated.
	#Those results are the inputs of backpropagation() function which determine the delta Weights and delta biases.
	#And after update those wights.
	#NOTE in the first try we use batch size = 1
	def train(self, x, y, batchSize=1, numberOfEpochs=100, lr = 0.001):
		numberOfAllImages = len(x)
		for epoch in range(numberOfEpochs):
			totalLoss = 0
			numberOfBatches = int(np.ceil(numberOfAllImages / batchSize))
			print(numberOfBatches)
			for batch in range(numberOfBatches):
				if batch == numberOfBatches:
					batchSize = (numberOfAllImages - ((batch - 1) * batchSize))
					batchX = x[batch:batch+batchSize]
					batchY = y[batch:batch+batchSize]
				else:
					batchX = x[batch:batch+batchSize]
					batchY = y[batch:batch+batchSize]
				prediction, cacheList = self.forward(batchX)
				dW, dB, loss = self.backpropagation(prediction, batchY, cacheList)
				self.doUpdateWeightsAndBiases(dW, dB, lr)
				totalLoss += loss
			cost = (totalLoss/numberOfBatches)
			print("[", epoch, "] epoch ", "average cost ",  cost)