#!/usr/bin/env python

# TODO
#
# @author Nagy Marton
# @author Kisházi "janohhank" János

# Scientific computing library.
import numpy

'''
'' Neural network layer abstraction class which can be a convolutional layer or a fully connected layer.
'' Initialize the weights of the kernels and the bias vectors elements.
''
'' TODO This class need to move into another script (Layer.py)
''
'' @author Nagy Marton
'' @author Kisházi "janohhank" János
'''
class Layer:
	# The layer sequence number of neural network.
	layerID = None

	# The layer type, it can be Convolution or FullyConnected.
	layerType = None

	# The layer activation (nonlinearity) function type.
	activationType = None

	# Stored kernels weights map.
	weights = {}

	# Stored bias vectors map.
	biases = {}

	'''
	'' Initialiaze a neural network layer elements and weights, biases.
	'' @param layerID, is the identical number of the layer.
	'' @param initializationType, is the initialization function type of the weights and biases.
	'' @param layerType, is the layer type, it can be Convolution and FullyConnected.
	'' @param activationType, is the nonlinearity function type which contains the lyer.
	'' @param weightSize, is the layer structure, the convention is when the layer convulutional [N,H,W,C], [K, C] when is fully connected.
	'''
	def __init__(
		self,
		layerID,
		initializationType,
		layerType,
		activationType,
		weightSize
	):
		self.layerID = layerID
		self.layerType = layerType
		self.activationType = activationType

		if(self.layerType == "Convolution"):
			numberOfKernels = weightSize[0]
			kernelHeight = weightSize[1]
			kernelWidth = weightSize[2]
			channels = weightSize[3]

			# TODO put initialization into a function
			if(initializationType == "UniformRandom"):
				for kernelIdx in range(numberOfKernels):
					self.weights[kernelIdx] = numpy.random.uniform(-0.01, +0.01, (kernelHeight, kernelWidth, channels))
					self.weights[kernelIdx]= self.weights[kernelIdx].astype(dtype=numpy.float128)
					self.biases[kernelIdx] = numpy.random.uniform(-0.01, +0.01, (1))
					self.biases[kernelIdx]= self.biases[kernelIdx].astype(dtype=numpy.float128)
			else:
				raise Exception("Unhandled initialization type: " + str(initializationType))
		elif(self.layerType == "FullyConnected"):
			# TODO it must calculate instead of arbitrary give it.
			lastFeatureMapSize = weightSize[0]
			numberOfOutPut = weightSize[1]

			# TODO put initialization into a function
			if(initializationType == "UniformRandom"):
				self.weights[0] = numpy.random.uniform(-0.01, +0.01, (lastFeatureMapSize, numberOfOutPut))
				self.weights[0]= self.weights[0].astype(dtype=numpy.float128)
				self.biases[0] = numpy.random.uniform(-0.01, +0.01, (numberOfOutPut))
				self.biases[0]= self.biases[0].astype(dtype=numpy.float128)
			else:
				raise Exception("Unhandled initialization type: " + str(initializationType))
		else:
			raise Exception("Unhandled layer type: " + str(layerType))

'''
'' Representation a type of convulutional neural network and contains all the
'' possible operations, for example forward and backpropagation.
'' It stores the neural network layers in a list.
''
'' @author Nagy Marton
'' @author Kisházi "janohhank" János
'''
class CNN:
	# Neural network layers storage.
	layers = []

	'''
	'' Initialize a neural network.
	'' @param initializationType, is the initialization function type.
	'' @param layerTypeList, is the layers type of each layer.
	'' @param activationTypeList, is the nonlinearity type of each layers.
	'' @param weightSizeList, is each layer structure, by the convention is when convolutional [N,H,W,C], and  [K, C] when its fully connected.
	'''
	def __init__(
		self,
		initializationType,
		layerTypeList,
		activationTypeList,
		weightSizeList
	):
		# All lists has to be the same number of elements.
		numberOfLayers = len(layerTypeList)
		for layerID in range(numberOfLayers):
			self.layers.append(
				Layer(
					layer,
					initializationType,
					layerTypeList[layerID],
					activationTypeList[layerID],
					weightSizeList[layerID]
				)
			)

	'''
	'' Propagates the information forward to the output of CNN.
	'' Creates a cache list which contains all usefull results during forward run of CNN.
	'' For example if we want to find the 6th layer weights.lists, it seems like: cacheList[5]["Weights"]
	'' @param X, is the input images list.
	'''
	def forward(self, X):
		predictions = [None] * len(X)
		# TODO we did not handle the batch list yet. The cacheList is not enough in this form if we use batch load in.
		for img in range(len(X)):
			# First input.
			output = X[img]

			cacheList = []
			numberOfLAyers = len(self.layers)
			for layerID in range(numberOfLAyers):
				if(self.layers[layerID].layerType == "Convolution"):
					cacheMap = {}

					#print("BEFORE Padding:", output)
					output, cache = self.doPaddingOnLayer(output, layerID)
					cacheMap["Padding"] = cache
					#print("AFTER Padding:", output)

					#print("BEFORE CONV:", output)
					output, cache = self.doConvolutionsOnLayer(output, layerID)
					cacheMap["X"] = cache[0]
					cacheMap["Weights"] = cache[1]
					#print("AFTER CONV:", output)

					#print("BEFORE CONV+bias:", output)
					output, cache = self.doBiasAddOnLayer(output, layerID)
					cacheMap["Biases"] = cache
					cacheMap["Z"] = output # "Z" is the logits before activation function.
					#print("AFTER CONV+bias:", output)

					#print("BEFORE ACTIVATION:", output)
					output = self.doActivationFunctionOnLayer(output, layerID)
					cacheMap["Activations"] = output
					#print("AFTER ACTIVATION:", output)

					#print("BEFORE MaxPool:", output)
					output, cache = self.doMaxPoolingOnlayer(output, 2, 2, 2, 2)
					cacheMap["PoolParameters"] = cache
					#print("AFTER MaxPool:", output)

					cacheList.append(cacheMap)
				# WARNING fully connected before a convolutional layer is not working, check this.
				elif(self.layers[layerID].layerType == "FullyConnected"):
					cacheMap = {}

					#print("BEFORE FC:", output)
					output, cache = self.doFullyConnectedOperationOnLayer(output, layerID)
					cacheMap["X"] = cache[0]
					cacheMap["Shape"] = cache[1]
					cacheMap["Weights"] = cache[2]
					cacheMap["Biases"] = cache[3]
					cacheMap["Z"] = output
					#print("AFTER FC:", output)

					#print("BEFORE ACTIVATION:", output)
					output = self.doActivationFunctionOnLayer(output, layerID)
					#print("AFTER ACTIVATION:", output)

					cacheList.append(cacheMap)
			predictions[img] = output
		return predictions, cacheList

	'''
	'' If we want to get same output sizes after the convolution, like it were before
	'' this function determines the those parameters of padding. (xLeft, xRight, yBot, yTop)
	'' Note that this is a naive function only works on the symmetric input and symmetric kernels.
	'' @param inputFeatureMap, is the input feature mapv (3D tensor).
	'' @param layerID, is the current layer ID.
	'''
	def doPaddingParameterDetermination(self, inputFeatureMap, layerID):
		# The base equation of the size of next feature map is:
		#		O = ((I - K + 2P) / S) + 1
		#	where
		#		O: output size
		#		I: input size
		#		K: kernel size
		#		P: pading size
		#		S: kernel stride
		# If we want to same size we need to I = O and we sort the equation to P.
		#		P = (I*S - S - I + K) / 2

		(fH,fW,fC) = self.layers[layerID].weights[0].shape

		# NOTE we handle only symmetric kernels now, later it will be different aspect ratio too.
		kernelSizes = fH

		#NOTE we use one kernel strides, later it will be a parameters too.
		kernelStride = 1

		shapeOfFeatureMap = inputFeatureMap.shape
		P = int(
			numpy.ceil(
				((shapeOfFeatureMap[0] * kernelStride) - kernelStride - shapeOfFeatureMap[0] + kernelSizes) / 2
			)
		)

		# Return P in which is equal with all parameters (xLeft, xRight, yBot, yTop) of padding.
		return P


	'''
	'' Creates a padding around the feature map. In this case padding means fill the map boundies with zero elements.
	'' @param inputFeatureMap, is the input feature map (3D tensor).
	'' @param layerID, is the current layer ID.
	'''
	def doPaddingOnLayer(self, inputFeatureMap, layerID):
		# Call a naive parameter determinator function.
		padding = self.doPaddingParameterDetermination(inputFeatureMap, layerID)

		# Because we assumes the symmetry all parameter is the same.
		yBot = yTop = xLeft = xRight = padding

		# Handle different to the opposite sides of the tensor, because the kernel sizes can be even
		# and if the feature map is not even it can be indexing out from the input.
		# Assymetric kernel sizes make it neccessary of the neighboor sides should be different have different shapes at padding.
		shapeOfFeatureMap = inputFeatureMap.shape
		H = shapeOfFeatureMap[0]
		W = shapeOfFeatureMap[1]
		C = shapeOfFeatureMap[2]
		
		newH = H + yBot + yTop
		newW = W + xLeft + xRight
		outputFeatureMap = numpy.zeros((newH, newW, C), dtype=numpy.float128)
		outputFeatureMap[yTop:-yBot, xLeft:-xRight, :] = inputFeatureMap

		cache = padding

		return outputFeatureMap, cache

	'''
	'' Does a convolution on all kernels at a layer.
	''
	'' @param inputFeatureMap, is the input feature map (3D tensor).
	'' @param layerID, is the current layer ID.
	'' @return The output of this function is a feature map.
	''	It fills a cache variable with inputs and weigths,
	''	because at the backpropagation it will use for further error determination.
	'''
	def doConvolutionsOnLayer(self, inputFeatureMap, layerID):
		# We assumes that all kernels are the same shape.
		(fH,fW,fC) = self.layers[layerID].weights[0].shape

		# Feature map size.
		outputDepth = len(self.layers[layerID].weights)
		(inputWidth, inputHeight, inputChannels) = inputFeatureMap.shape
		outputHeight = inputHeight - fH + 1
		outputWidth = inputWidth - fW + 1

		outputFeatureMap = numpy.zeros((outputHeight, outputWidth, outputDepth), dtype=numpy.float128)
		for kernelIdx in range(outputDepth):
			for h in range(outputHeight):
				for w in range(outputWidth):
					xSlice = inputFeatureMap[h:h+fH, w:w+fW, :]
					outputFeatureMap[h, w, kernelIdx] = numpy.sum(xSlice * self.layers[layerID].weights[kernelIdx])

		# For the backpropagation we need to store the input and weight tensors.
		cache = (inputFeatureMap.copy(), self.layers[layerID].weights.copy())

		return outputFeatureMap, cache

	'''
	'' Adds biases to all elements of the input.
	'' @param inputFeatureMap, is the input feature map (3D tensor).
	'' @param layerID, is the current layer ID.
	'''
	def doBiasAddOnLayer(self, inputFeatureMap, layerID):
		outputFeatureMap = numpy.zeros_like(inputFeatureMap, dtype=numpy.float128)

		outputDepth = len(self.layers[layerID].biases)
		for kernelIdx in range(outputDepth):
			outputFeatureMap[:,:,kernelIdx] = inputFeatureMdoMaxPoolingOnlayerap[:,:,kernelIdx] + self.layers[layerID].biases[kernelIdx]

		# For the backpropagation we need to store the input and bias vectors.
		cache = self.layers[layerID].biases.copy()

		return outputFeatureMap, cache

	'''
	'' Use the activation function (nonlinearity funtcion) on a layer.
	'' @param inputFeatureMap, is the input feature map (3D tensor).
	'' @param layerID, is the current layer ID.
	'''
	def doActivationFunctionOnLayer(self, inputFeatureMap, layerID):
		typeOfNonlinearity = self.layers[layerID].activationType

		# Gets the nonlinearity function.
		activationFunction = self.getActivationFunction(typeOfNonlinearity)

		# Use activation function on input feature map.
		outputFeatureMap = activationFunction(inputFeatureMap)

		return outputFeatureMap

	'''
	'' Does a max-pooling on a layer.
	'' Reduces the height and width of feature map, (select the maximal activity in the kernel slices).
	'' @param inputFeatureMap, is the input feature map (3D tensor).
	'' @param kernelHeight, TODO stores this in the Layer.
	'' @param kernelWidth, TODO stores this in the Layer.
	'' @param strideY, TODO stores this in the Layer.
	'' @param strideX, TODO stores this in the Layer.
	'''
	def doMaxPoolingOnlayer(self, inputFeatureMap, kernelHeight, kernelWidth, strideY, strideX):
		(inputWidth, inputHeight, inputChannels) = inputFeatureMap.shape

		# Determine the output resolution after pooling.
		outputHeight = int(numpy.ceil((inputHeight - kernelHeight) / strideY) + 1)
		outputWidth = int(numpy.ceil((inputWidth - kernelWidth) / strideX) + 1)

		outputFeatureMap = numpy.zeros((outputHeight, outputWidth, inputChannels), dtype=numpy.float128)
		for c in range(inputChannels):
			for h in range(outputHeight):
				for w in range(outputWidth):
					# TODO indexing out from input we should handle those foreign pixels.
					xSlice = inputFeatureMap[(h*strideY):(h*strideY)+kernelHeight, (w*strideX):(w*strideX)+kernelWidth, c]
					outputFeatureMap[h, w, c] = numpy.max(xSlice)

		parametersOfPooling = {}
		parametersOfPooling["stride"] = strideY #NOTE at this time this is the same than strideX.
		parametersOfPooling["poolingFilterHeight"] = kernelHeight
		parametersOfPooling["poolingFilterWidth"] = kernelWidth

		# For the backpropagation we need to store the input the parameters of pooling.
		cache = parametersOfPooling

		return outputFeatureMap, cache

	'''
	'' Does a fully connected tensor operation on layer.
	'' This function multiplies the input activation (1D) with the weights of current layer (2D) and add to that the biases (1D).
	'' @param inputFeatureMap, is the input feature map (3D tensor).
	'' @param layerID, is the current layer ID.
	'''
	def doFullyConnectedOperationOnLayer(self, inputFeatureMap, layerID):
		# Create 1D vector from 3D feature map, the flatten() do it if inputFeatureMap is a numpy.array.
		xShape = inputFeatureMap.shape
		X = numpy.reshape(inputFeatureMap.copy(), (xShape[0] * xShape[1] * xShape[2]))

		(countOfActivation, countOfClasses) = self.layers[layerID].weights[0].shape

		outputVector = numpy.add(numpy.matmul(X, self.layers[layerID].weights[0]), self.layers[layerID].biases[0])

		# For the backpropagation we need to store the input and weights and biases.
		cache = (X, xShape, self.layers[layerID].weights.copy(), self.layers[layerID].biases.copy())

		return outputVector, cache

####################################### janohhank rework end line ######################################
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
		
		dX = numpy.zeros(X.shape, dtype=numpy.float128)
		
		#Feature map sizes
		countOfKernels = len(self.layers[layerID].weights)
		for kernelIdx in range(countOfKernels):
			#dE/dW
			dW = numpy.zeros_like(self.layers[layerID].weights[kernelIdx], dtype=numpy.float128)	#NOTE: If all kernels are same this line can goto out from this for cycle.
			#dE/dB
			dB = numpy.zeros_like(self.layers[layerID].biases[kernelIdx], dtype=numpy.float128)	#NOTE: If all kernels are same this line can goto out from this for cycle.
			for h in range(Hcurr):
				for w in range(Wcurr):
					dX[h:h+fH, w:w+fW, :] += W[kernelIdx] * delta[h,w,kernelIdx]	#Only thos elementsof dX accumulates, which contributed to delta[h,w,kernelIdx] elements.
					dW += X[h:h+fH, w:w+fW, :] * delta[h,w,kernelIdx]
					dB += delta[h,w,kernelIdx]
			dWList.append(dW)
			dBList.append(dB)
		
		
		return dX, dWList, dBList

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
		
		dW = numpy.outer(X, delta)
		dW= dW.astype(dtype=numpy.float128)
		dB = delta
		dB= dB.astype(dtype=numpy.float128)
		
		dX = numpy.dot(W[0], delta)
		dX= dX.astype(dtype=numpy.float128)
		
		dWList.append(dW)
		dBList.append(dB)
		
		return dX, dWList, dBList

########################################################################################################################
##Activations(/Nonlinearities) and their derivates (derivates are neccessary during backpropagation).
	
	#This function contains same activation functions then getActivationFunction()[below], but this function is not use lambdas.
	def getActivationFunctionWithoutLambda(self, name, X):
		if(name == "sigmoid"):
			y = numpy.copy(X)
			results = numpy.exp(y)/(1+numpy.exp(y))
			return results
		elif(name == "linear"):
			return X
		elif(name == "relu"):
			y = numpy.copy(X)
			y[y<0] = 0
			return y
		elif(name == "softmax"):
			xx = numpy.copy(x)
			y = numpy.exp(xx) / numpy.sum(numpy.exp(xx))
			return y
		else:
			print('Unknown activation function. linear is used')
			return lambda x: x
		return

	
	#This function return the selected activation function.
	#It is called by doActivationFunctionOnLayer() function.
	def getActivationFunction(self, name):
		if(name == "sigmoid"):
			return lambda x : numpy.exp(x)/(1+numpy.exp(x))
		elif(name == "linear"):
			return lambda x : x
		elif(name == "relu"):
			def relu(x):
				y = numpy.copy(x)
				y[y<0] = 0
				return y
			return relu
		elif(name == "softmax"):
			def softmax(x):
				xx = numpy.copy(x)
				y = numpy.exp(xx) / numpy.sum(numpy.exp(xx))
				return y
			return softmax
				
		else:
			print('Unknown activation function. linear is used')
			return lambda x: x
    
    
	#At the backpropagation we should use these functions instead of the original activation functions.
	#These are the derivatives of the original ones.
	def getDerivitiveActivationFunction(self, name):
		if(name == "sigmoid"):
			sig = lambda x : numpy.exp(x)/(1+numpy.exp(x))
			return lambda x :sig(x)*(1-sig(x))
		elif(name == "linear"):
			return lambda x: 1
		elif(name == "relu"):
			def relu_diff(x):
				y = numpy.copy(x)
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
	#It is a naive implementation of backward pooling based on:
	#https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/pooling_layer.html
	def doMaxPoolingOnlayerBackWard(self, pooledDeltaX, cache, xShape=None):
		
		#shape of the output feature map. This is the backpropagated error tensor.
		#It contain the error of next feature map and the derivates of the nonlinearity(wrt the logits)
		if not (xShape is None):
			#if the delta come from a fully connected it is a 1D vector
			#Reshape it into 3D tensor
			pooledDeltaX = numpy.reshape(pooledDeltaX, (xShape))
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

		unPooledDeltaX = numpy.zeros_like(unpooledX, dtype=numpy.float128)
		
		#for n in range(N):
		for c in range(C):
			for y in range(HH):
				for x in range(WW):
					xSlice = unpooledX[y*stride:y*stride+fHeight, x*stride:x*stride+fWidth, c]
					xSlice = numpy.squeeze(xSlice)
					#Select thos i,j which is (/are) the maximal values of input, we propagates only those errors.
					selectMax = numpy.where(xSlice >= numpy.amax(xSlice), 1, 0)
					
					unPooledDeltaX[y*stride:y*stride+fHeight, x*stride:x*stride+fWidth, c] = selectMax * pooledDeltaX[y,x,c]
		return unPooledDeltaX

########################################################################################################################
##Loss function
	#This loss function compute the absolute value of loss
	def lossFunction(self, output, target):
		#target is a scalaer betweeen 0 and 9
		#we transfrom this scalar into a binary vector with 10 elements.
		targetVector = numpy.zeros(len(output), dtype=numpy.float128)
		targetVector[target-1] = 1.0
		print(output)
		print(targetVector)

		loss = numpy.sum(numpy.absolute(numpy.subtract(output, targetVector)))
		return loss

########################################################################################################################
##Gradients of Weights and Biases Summarizer
	#This function sume up dWs and dBs for images (in a batch)
	def doWeightsAndBiasSum(self, dWAll, dBAll, dWImg ,dBImg):
		numberOfLayers = len(self.layers)
		for layerID in range(numberOfLayers):
			kernelCounts = len(self.layers[layerID].weights)
			for kernelIdx in range(kernelCounts):
				dWAll[layerID][kernelIdx] = numpy.add(dWAll[layerID][kernelIdx], dWImg[layerID][kernelIdx])
				dBAll[layerID][kernelIdx] = numpy.add(dBAll[layerID][kernelIdx], dBImg[layerID][kernelIdx])
		return dWAll, dBAll


########################################################################################################################
#Forward and Backward





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
				self.layers[layerID].weights[kernelIdx] = numpy.subtract(self.layers[layerID].weights[kernelIdx], (lr * dW[layerID][kernelIdx]))
				self.layers[layerID].biases[kernelIdx] = numpy.subtract(self.layers[layerID].biases[kernelIdx], (lr * dB[layerID][kernelIdx]))


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
			numberOfBatches = int(numpy.ceil(numberOfAllImages / batchSize))
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