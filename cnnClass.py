#!/usr/bin/python3
#
# This python contains the possible layers of a CNN and their backpropagation functions.
# Connect those layers in a CNN class.
#
# author Nagy Marton

import numpy as np


class Layer:
	def __init__(self, layrID, initializationType, layerType, activationType, weightSize):
		self.layrID = layrID
		self.initializationType = initializationType		#only Xavier allowed yet (WARNING the implementation is not Xavier yet.)
		self.layerType = layerType				#convolution or fully connected
		self.activationType = activationType			#nonlinearity of layer
		self.weights = {}
		self.biases = {}
		#TODO: The Xavier is not uniform between -1 and +1  fix it.
		if self.initializationType == "Xavier":
			if self.layerType == "Convolution":
				#NOTE: [N,H,W,C] is the convention.
				numberOfKernels = weightSize[0]
				kernelHeight = weightSize[1]
				kernelWidth = weightSize[2]
				channels = weightSize[3]
				for kernelIdx in range(numberOfKernels):
					self.weights[kernelIdx] = np.random.uniform(-1, +1, (kernelHeight, kernelWidth, channels))
					self.biases[kernelIdx] = np.random.uniform(-1, +1, (1))

			elif self.layerType == "FullyConnected":
				lastFeatureMapSize = weightSize[0]	#TODO it must calculate instead of arbitrary give it.
				numberOfOutPut = weightSize[1]
				self.weights[0] = np.random.uniform(-1, +1, (lastFeatureMapSize, numberOfOutPut))
				self.biases[0] = np.random.uniform(-1, +1, (numberOfOutPut))
		else:
			print("initializationType =", self.initializationType)
			raise Exception("This initializationType is not defined yet. Only Xavier is allowed.")

	def __del__(self): 
		print('Destructor called, Layer deleted.')


class CNN:
	layers = []
	#TODO: weightSizeList is list of lists, which list ith elements represents the shape of ith layer weights. (TODO create a function which create this lists from input parameters.)
	def __init__(self, initializationType, layerTypeList, activationTypeList, weightSizeList):
		numberOfLayers = len(layerTypeList)	#all lists in the arguments of this function have same number of elements.
		for layer in range(numberOfLayers):
			self.layers.append(Layer(layer, initializationType, layerTypeList[layer], activationTypeList[layer], weightSizeList[layer]))
	
	def __del__(self): 
		print('Destructor called, CNN deleted.')


########################################################################################################################
##Padding
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
		
		P = np.ceil(((inputFeatureMap * kernelStride) - kernelStride - inputFeatureMap + kernelSizes) / 2)

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
		(H, W, C) = inputFeatureMap.shape()
		newH = H + yBot + yTop
		newW = W + xLeft + xRight
		outputFeatureMap = np.zeros((newH, newW, C))
		outputFeatureMap[yTop:-yBot, xLeft:-xRight, :]	#WARNING: these function does not handle the batchsize yet.

		return outputFeatureMap

########################################################################################################################
##Conv and FC
	#Convolutional layer fowrard.
	def doConvolutionsOnLayer(self, inputFeatureMap, layerID):
		#Filter/Kernel sizes
		(fH,fW,fC) = self.layers[layerID].weights[0].shape
		
		#Feature map sizes
		outputDepth = len(self.layers[layerID])
		(inputWidth, inputHeight, inputChannels) = inputFeatureMap.shape
		outputHeight = inputHeight - fH + 1
		outputWidth = inputWidth - fW + 1
		
		outputFeatureMap = np.zeros((outputHeight, outputWidth, outputDepth))
		
		for kernelIdx in range(outputDepth):
			for h in range(outputHeight):
				for w in range(outputWidth):
					xSlice = inputFeatureMap[h:h+fH, w:w+fW, :]
					outputFeatureMap[h, w, kernelIdx] = np.sum(xSlice * self.layers[layerID].weights[kernelIdx])
		
		#for backpropagation we need the X and W tensors
		cache = (inputFeatureMap, self.layers[layerID].weights.copy())
		
		return outputFeatureMap, cache

	#Backward convoltuion
	def backwardConvoltuion(self, layerID, delta, cache, xShape=None):
		
		#Stored feature map
		X = cache["X"]
		
		#Stored wights
		W = cache["Weights"]
		
		#shape (H,W,C) of input
		(Hprev, Wprev, Cprev) = X.shape
		
		#shape (H,W,C) of a weight kernel
		(fH, fW, fC) = W[0].shape
		
		#shape of the output feature map. This is the backpropagated error tensor.
		#It contain the error of next feature map and the derivates of the nonlinearity(wrt the logits)
		if not (xShape is None):
			#if the delta come from a fully connected it is a 1D vector
			#Reshape it into 3D tensor
			delta.reshape=(xShape)
			(Hcurr, Wcurr, Ccurr) = delta.shape
		else:
			#The delta come from convoluton
			(Hcurr, Wcurr, Ccurr) = delta.shape
		
		#These list will be store the derivates of the weights and biases.
		dWList = []
		dBList = []
		
		dX = np.zeros(X.shape)
		
		#Feature map sizes
		countOfKernels = len(self.layers[layerID])
		for kernelIdx in range(countOfKernels):
			#dE/dW
			dW = np.zeros(self.layers[layerID].weights[kernelIdx].shape)	#NOTE: If all kernels are same this line can goto out from this for cycle.
			#dE/dB
			dB = np.zeros(self.layers[layerID].biases[kernelIdx].shape)	#NOTE: If all kernels are same this line can goto out from this for cycle.
			for h in range(Hcurr):
				for w in range(Wcurr):
					dX[h:h+fH, w:w+fW, :] += W[kernelIdx] * delta[h,w,kernelIdx]
					dW += X[h:h+fH, w:w+fW, :] * delta[h,w,kernelIdx]
					dB += delta[h,w,kernelIdx]
			dWList.append(dW)
			dBList.append(dB)
		
		
		return dX, dWList, dBList



	#Fully Connected layer forward.
	def doFullyConnectedOperationOnLayer(self, inputFeatureMap, layerID):
		#Create 1D vector from 3D feature map
		#.flatten() do it if inputFeatureMap is a numpy.array
		X = inputFeatureMap.copy()
		xShape = X.shape
		X.flatten()

		(countOfActivation, countOfClasses) = self.layer[layerID].weights[0].shape
		
		outputVector = [None] * countOfClasses
		for cls in range(countOfClasses):
			for act in range(countOfActivation):
				outputVector[cls] += np.add(np.multiply(X, self.layers[layerID].weights[0]), self.layers[layerID].biases[0])
		
		#for backpropagation we need the X and W tensors
		cache = (X, xShape, self.layers[layerID].weights.copy(), self.layers[layerID].biases.copy())
		
		return outputVector, cache

	#Backward Fully Conected layer
	def backwardFullyConnected(self, layerID, delta, cache):
		
		#Stored feature map (it is an 1D)
		X = cache["X"]
		#The shape of X befor flatten() xShape = (H,W,C)
		xShape = cache["Shape"]
		#Stored wights
		W = cache["Weights"]
		
		#shape (H,W,C) of input
		(Hprev, Wprev, Cprev) = xShape
		
		#shape (activation_pixels, output_classes) of a weight kernel
		(inputCount, outputCount) = W[0].shape
		
		#shape of the output feature map. This is the backpropagated error tensor.
		#It contain the error of next feature map and the derivates of the nonlinearity(wrt the logits)
		(Hcurr, Wcurr, Ccurr) = delta.shape
		
		#These list will be store the derivates of the weights and biases.
		dWList = []
		dBList = []
		
		#Feature map sizes
		countOfKernels = len(self.layers[layerID])
		
		dX 
		
		return dX, dWList, dBList


########################################################################################################################
##Bias
	#add biases to the input all elements
	def doBiasAddOnLayer(self, inputFeatureMap, layerID):
		#Feature map sizes
		outputDepth = len(self.layers[layerID])
		for kernelIdx in range(outputDepth):
			outputFeatureMap[:,:,kernelIdx] = inputFeatureMap[:,:,kernelIdx] + self.layers[layerID].biases[kernelIdx]
		
		#for backpropagation we need the X and W tensors
		cache = self.layers[layerID].biases.copy()
		
		#NOTE: the outputFeatureMap will store in the cacheList because it is the logits (commom notation is z)
		return outputFeatureMap, cache
    
    
##Activations(/Nonlinearities) and their derivates (derivates are neccessary during backpropagation).
    #This function return the selected activation function.
    #It is called in activationFunction() function.
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
	#It is waiting the results of conv + bias and return with the activation of the selected nonlinearity type.
	def doActivationFunctionOnLayer(self, inputFeatureMap, typeOfNonlinearity):
		
		#Select nonlinearity function
		activationFunction = self.getActivationFunction(typeOfNonlinearity)
		
		#Use activation function on input feature map
		outputFeatureMap = activationFunction(inputFeatureMap)

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
			return lambda x :softmax(x)*(1-softmax(x))		#NOTE: the description is different from this.
		else:
			print('Unknown activation function. linear is used')
			return lambda x: 1

	#TODO doDerivatActivation...
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
		outputHeight = np.ceil((inputHeight - kernelHeight) / strideY) + 1
		outputWidth = np.ceil((inputWidth - kernelWidth) / strideX) + 1
		
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
	def doMaxPoolingOnlayerBackWard(self, pooledDeltaX, cache):
		unpooledX = cache["Activations"]
		parametersOfPooling = cache["PoolParameters"]
		
		#N is the batchSize.
		#H, W, C is the height width channels of feature map which was pooled.
		#N, H, W, C = unpooledX.shape()
		H, W, C = unpooledX.shape()
		
		stride = parametersOfPooling["stride"]
		fHeight = parametersOfPooling["poolingFilterHeight"]
		fWidth = parametersOfPooling["poolingFilterWidth"]
		
		#NOTE: NN = N and CC = C, because pool decrease only the width and height of feature map
		#NN, HH, WW, CC = pooledDeltaX.shape()
		HH, WW, CC = pooledDeltaX.shape()
		
		unPooledDeltaX = None
		unPooledDeltaX = np.zeros(unpooledX)
		
		#for n in range(N):
		for depth in range(C):
			for y in range(HH):
				for x in range(WW):
					xSlice = unpooledX[y*stride:y*stride+fHeight, x*stride:x*stride+fWidth, c]
					
					#Select thos i,j which is (/are) the maximal values of input, we propagates only those errors.
					selectMax = (xSlice == np.max(xSlice))
					
					unPooledDeltaX[y*stride:y*stride+fHeight, x*stride:x*stride+fWidth, c] = selectMax * pooledDeltaX[y,x,depth]
		return unPooledDeltaX

########################################################################################################################
##Loss function
	#This loss function compute the absolute value of loss
	def lossFunction(self, output, target):
		loss = np.sum(np.absolute(np.subtract(output, target)))
		return loss


########################################################################################################################
#Forward and Backward
	#This function get the input and propagate the information forward to the output of CNN.
	#It is a pipeline. The output of each function is the input of the next one.
	#A for cycle iterate over all layers.
	def forward(self, X):
		cacheList = []
		#cacheList[5]["Weights"]
		#TODO reshape X if it is neccessary

		#first input
		#it called output because later we will call the functions with these name.
		output = X

		#For cycle to create the cnn pipeline.
		numberOfLAyers = len(self.layers)
		for layerID in range(numberOfLAyers):
			if self.layers[layerID].layerType == "Convolution":
				cacheMap = {}
				output = self.doPaddingOnLayer(output, layerID)
				output, cache = self.doConvolutionsOnLayer(output, layerID)
				cacheMap["X"] = cache[0]
				cacheMap["Weights"] = cache[1]
				output, cache = self.doBiasAddOnLayer(output, layerID)
				cacheMap["Biases"] = cache
				cacheMap["Z"] = output							#It is the logits before activation function.
				output = self.doActivationFunctionOnLayer(output, self.layers[layerID].activationType)
				cacheMap["Activations"] = output
				output, cache = self.maxPooling2D(output, 2, 2, 2, 2)
				cacheMap["PoolParameters"] = cache[0]
				cacheList.append(cacheMap)
			elif self.layers[layerID].layerType == "FullyConnected":			#WARNING: fully connected before a convolutional layer is not working, check this.
				output, cache = self.doFullyConnectedOperationOnLayer(output, layerID)
				cacheMap["X"] = cache[0]
				cacheMap["Shape"] = output[1]
				cacheMap["Weights"] = cache[2]
				cacheMap["Biases"] = cache[3]
				cacheMap["Z"] = output
				cacheList.append(cacheMap)
				output = self.doActivationFunctionOnLayer(output, self.layers[layerID].activationType)
			
			
		return output, cacheList




	#backpropagation
	def backpropagation(self, output, target, cacheList):
		
		loss = lossFunction(output, target)
		
		numberOfLayers = len(self.layers)
		
		#deltas = dE/dZ  it is the error for each layer
		deltas = [None] * numberOfLayers
		deltas[-1] = (loss)*(self.doDerivateOfActivationFunctionOnLayer(self.layers[-1].activationType))(cacheList[-1]["Z"])
		
		dWAll = []
		dBAll = []
		
		for layerID in reversed(range(numberOfLayers - 1)):
			if self.layers[layerID].layerType == "Convolution":
				#Check that the following layer after convolution is a fully connected
				if self.layers[layerID + 1].layerType == "FullyConnected":
					xShape = cacheList[layerID]["Shape"]
					dXLayer, dWLayer, dBLayer = self.backwardConvoltuion(layerID, deltas[layerID+1], cacheList[layerID], xShape)
				else:
					dXLayer, dWLayer, dBLayer = self.backwardConvoltuion(layerID, deltas[layerID+1], cacheList[layerID])
				#dX and the derivates of the nonlinearity have same shapes.
				dXLayer = self.doMaxPoolingOnlayerBackWard(dXLayer, cacheList[layerID])
				deltas[layerID] = dXLayer * (self.doDerivateOfActivationFunctionOnLayer(self.layers[layerID].activationType))(cacheList[layerID]["Z"])
				dWAll.append(dWLayer)
				dBAll.append(dBLayer)
			elif self.layers[layerID].layerType == "FullyConnected":
				dXLayer, dWLayer, dBLayer = self.backwardFullyConnected(layerID, deltas[layerID+1], cacheList[layerID])
				#dX and the derivates of the nonlinearity have same shapes.
				deltas[layerID] = dXLayer * (self.doDerivateOfActivationFunctionOnLayer(self.layers[layerID].activationType))(cacheList[layerID]["Z"])
				dWAll.append(dWLayer)
				dBAll.append(dBLayer)
				
		return dWAll, dBAll, loss

########################################################################################################################
##Weights and Biases UPDATES
	#This function updates the weights and biases depending on learning rate and determined dW and dB.
	def doUpdateWeightsAndBiases(self, dW, dB, lr):
		numberOfLayers = len(self.layers)
		for layerID in range(numberOfLayers):
			kernelCounts = len(self.layers[layerID])
			for kernelIdx in range(kernelCounts):
				self.layers[layerID].weights[kernelIdx] = np.subtract(self.layers[layerID].weights[kernelIdx], (lr * dW[layerID][kernelIdx]))
				self.layers[layerID].biases[kernelIdx] = np.subtract(self.layers[layerID].biases[kernelIdx], (lr * dB[layerID][biases]))


########################################################################################################################
##TRAIN
	#TODO in the first try we use batch size = 1
	def train(self, x, y, batchSize=1, numberOfEpochs=100, lr = 0.01):
		#TODO image load

		for epoch in range(numberOfEpochs):
			totalLoss = 0
			numberOfBatches = np.ceil(numberOfAllImages / batchSize)
			for batch in range(numberOfBatches):
				#TODO loadImage + loadLabel
				if batch == numberOfBatches:
					batchSize = (numberOfAllImages - ((batch - 1) * batchSize))
					batchX = imageLoader(sourceOfDB, batchSize, batch)
					batchY = labelLoader(sourceOfDB, batchSize, batch)
				else:
					batchX = imageLoader(sourceOfDB, batchSize, batch)
					batchY = labelLoader(sourceOfDB, batchSize, batch)
					
				prediction, cacheList = forward(batchX)
				dW, dB, loss = backpropagation(prediction, batchY, cacheList)
				self.doUpdateWeightsAndBiases(dW, dB, lr)
				totalLoss += loss
			cost = (totalLoss/numberOfBatches)
			print("[", epoch, "] epoch ", "average cost ",  cost)
		return cnn		#TODO ????what should it return????