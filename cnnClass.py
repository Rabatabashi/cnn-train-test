#!/usr/bin/python3
#
# This python contains the possible layers of a CNN and their backpropagation functions.
# Connect those layers in a CNN class.
#
# author Nagy Marton

import numpy as np


class Layer:
    def __init__(self, layrID, initializationType, layerType, weightSize):
        self.layrID = layrID
        self.initializationType = initializationType
        self.layerType = layerType
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
                lastFeatureMapSize = weightSize[0]
                numberOfOutPut = weightSize[1]
                self.weights[0] = np.random.uniform(-1, +1, (lastFeatureMapSize, numberOfOutPut))
                self.biases[0] = np.random.uniform(-1, +1, (numberOfOutPut))
        else:
            print("initializationType =", self.initializationType)
            raise Exception("This initializationType is not defined yet.")
        
    def __del__(self): 
        print('Destructor called, Layer deleted.')


class CNN:
	self.layers = []
	#TODO: weightSizeList is list of lists, which list ith elements represents the shape of ith layer weights. (TODO create a function which create this lists from input parameters.)
	def __init__(self, numberOfLayers, initializationType, layerTypeList, weightSizeList):
		for layer in range(numberOfLayers):
			self.layers.append(Layer(layer, initializationType, layerTypeList[layer], weightSizeList[layer]))
	
	def __del__(self): 
		print('Destructor called, CNN deleted.')



	#This function create a padding around the feature map.
	#TODO
	def padding():
		
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

    #Fully Connected layer forward.
    def doFullyConnectedOperationOnLayer(self, inputFeatureMap, layerID):
        #Create 1D vector from 3D feature map
        #.flatten() do it if inputFeatureMap is a numpy.array
        X = inputFeatureMap.copy()
        X.flatten()

        (countOfActivation, countOfClasses) = self.layer[layerID].weights[0].shape
        
        outputVector = [None] * countOfClasses
        for cls in range(countOfClasses):
            for act in range(countOfActivation):
                outputVector[cls] += np.add(np.multiply(X, self.layers[layerID].weights[0]), self.layers[layerID].biases[0]))
        
        #for backpropagation we need the X and W tensors
        cache = (X, self.layers[layerID])
        
        return outputVector, cache

########################################################################################################################
##Bias
    #add biases to the input all elements
    def doBiasAddOnLayer(self, inputFeatureMap, layerID):
	#TODO outputhdepth
	
	
        for kernelIdx in range(outputDepth):
            outputFeatureMap[:,:,kernelIdx] = inputFeatureMap[:,:,kernelIdx] + self.layers[layerID].biases[kernelIdx]
            
	#for backpropagation we need the X and W tensors
        cache = self.layers[layerID].biases.copy()
	
        return outputFeatureMap, cache
    
    
##Activations(/Nonlinearities) and their derivates (derivates are neccessary during backpropagation).
    #This function return the selected activation function.
    #It is called in activationFunction() function.
	def getActivationFunction(self, name):
		if(name == 'sigmoid'):
			return lambda x : np.exp(x)/(1+np.exp(x))
		elif(name == 'linear'):
			return lambda x : x
		elif(name == 'relu'):
			def relu(x):
				y = np.copy(x)
				y[y<0] = 0
				return y
			return relu
		elif(name == 'softmax'):
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
		if(name == 'sigmoid'):
			sig = lambda x : np.exp(x)/(1+np.exp(x))
			return lambda x :sig(x)*(1-sig(x)) 
		elif(name == 'linear'):
			return lambda x: 1
		elif(name == 'relu'):
			def relu_diff(x):
				y = np.copy(x)
				y[y>=0] = 1
				y[y<0] = 0
				return y
			return relu_diff
		elif(name == 'softmax'):
			softmax = self.getActivationFunction('softmax')
			
			return 
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
		#TODO 
		outputHeight = np.ceil((inputHeight - kernelHeight) / strideY) + 1
		outputWidth = np.ceil((inputWidth - kernelWidth) / strideX) + 1
		
		for c in range(inputChannels):
			for h in range(outputHeight):
				for w in range(outputWidth):
					#TODO indexing out from input we should handle those foreign pixels
					xSlice = inputFeatureMap[(h*strideY):(h*strideY)+kernelHeight, (w*strideX):(w*strideX)+kernelWidth, c]
					outputFeatureMap[h, w, c] = np.max(xSlice)
		
		return outputFeatureMap

	#It is a naive implementation of backward pooling based on:
	#https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/pooling_layer.html
	def doMaxPoolingOnlayerBackWard(self, pooledDeltaX, cache):
		
		unpooledX, parametersOfPooling = cache
		
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
        for layer in range(numberOfLAyers):
            if self.layers[layer].layerType == "convolutional":
		cacheMap = {}
                #TODO padding
                output, cache = self.doConvolutionsOnLayer(output, layer)
                cacheMap["X"] = cache[0]
                cacheMap["Weights"] = cache[1]
                
                
                output, cache = self.doBiasAddOnLayer(output, layer)
                cacheMap["Biases"] = cache
                output = self.doActivationFunctionOnLayer(output, typeOfNonlinearity)    #NOTE: typeOfNonlinearity is not defined
                cacheMap["Activations"] = output
                output = self.maxPooling2D(output, 2, 2, 2, 2)
                
                cacheList.append(cacheMap)
            elif self.layers[layer].layerType == "fullyConnected":              #WARNING: fully connected before a convolutional layer is not working, check this.
                output, cache = self.doFullyConnectedOperationOnLayer(output, layer)
                cacheList.append(cacheMap)
                output = self.doActivationFunctionOnLayer(output, typeOfNonlinearity)    #NOTE: typeOfNonlinearity is not defined
                
                
        return output, cacheList




    #backpropagation
    def backpropagation(self, output, target, cacheList):
        
        
        
        
        dw = []  # dC/dW
        db = []  # dC/dB
        deltas = [None] * len(self.weights)  # delta = dC/dZ  known as error for each layer
        # insert the last layer error
        deltas[-1] = ((y-a_s[-1])*(self.getDerivitiveActivationFunction(self.activations[-1]))(z_s[-1]))
        # Perform BackPropagation
        for i in reversed(range(len(deltas)-1)):
            deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*(self.getDerivitiveActivationFunction(self.activations[i])(z_s[i]))        
            batch_size = y.shape[1]
            db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas]
            dw = [d.dot(a_s[i].T)/float(batch_size) for i,d in enumerate(deltas)]
            # return the derivitives respect to weight matrix and biases
            return dw, db



	#TODO in the first try we use batch size = 1
    def train(self x, y, batchSize=1, numberOfEpochs=100, lr = 0.01):
        #TODO train
        #TODO image load
        
        for epoch in range(numberOfEpochs):
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
				
			sums, activations = forward(batchX)
			dW, dB = backpropagation(batchY, sums, activations)
                    
        
        


        
        
