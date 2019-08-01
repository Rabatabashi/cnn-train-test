#!/usr/bin/python3
#
# This python contains the possible layers of a CNN and their backpropagation functions.
# Connect those layers in a CNN class.
#
# author Nagy Marton

import numpy as np


class CNN:
    #TODO: weightSizeList is list of lists, which list ith elements represents the shape of ith layer weights. (TODO create a function which create this lists from input parameters.)
    def __init__(self, numberOfLayers, initializationType, layerTypeList, weightSizeList):
        self.layers = []
        for layer in range(numberOfLayers):
            self.layers.append(self.Layer(layer, initializationType, layerTypeList[layer], weightSizeList[layer]))
    
    def __del__(self): 
        print('Destructor called, CNN deleted.')

    #This function create a padding around the feature map.
    #TODO
    def padding():
        
        return outputFeatureMap

    def convolution(self, inputFeatureMap, layerID):
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
        cache = (inputFeatureMap, self.layers[layerID])
        
        return outputFeatureMap, cache

    def biasAdd(self, inputFeatureMap):
        for kernelIdx in range(outputDepth):
            outputFeatureMap[:,:,kernelIdx] = inputFeatureMap[:,:,kernelIdx] + self.layers[layerID].biases[kernelIdx]
        return outputFeatureMap

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
        else:
            print('Unknown activation function. linear is used')
            return lambda x: x

    #Nonlinearity or Activation function
    #It is waiting the results of conv + bias and return with the activation of the selected nonlinearity type.
    def activationFunction(self, inputFeatureMap, typeOfNonlinearity):
        
        #Select nonlinearity function
        activationFunction = self.getActivationFunction(typeOfNonlinearity)
        
        #Use activation function on input feature map
        outputFeatureMap = activationFunction(inputFeatureMap)
        
        return outputFeatureMap

    #Max-pooling layer - reduce the height and width of feature map, (select the maximal activity in the kernel slices)
    def maxPooling2D(self, inputFeatureMap, kernelHeight, kernelWidth, strideY, strideX):
        (inputWidth, inputHeight, inputChannels) = inputFeatureMap.shape
        #determine the output resolution after pooling
        #TODO 
        outputHeight = np.ceil((inputHeight - kernelHeight) / strideY) + 1
        outputWidth = np.ceil((inputWidth - kernelWidth) / strideX) + 1
        
        for c in range(inputChannels):
            for h in range(outputHeight):
                for w in range(outputWidth):
                    xSlice = inputFeatureMap[(h*strideY):(h*strideY)+kernelHeight, (w*strideX):(w*strideX)+kernelWidth, c]
                    outputFeatureMap[h, w, c] = np.max(xSlice)
        
        return outputFeatureMap

    #Fully Connected layer forward
    def fullyConnected(self, inputFeatureMap, layerID):
        #Create 1D vector from 3D feature map
        #.flatten() do it if inputFeatureMap is a numpy.array
        inputFeatureMap.flatten()

        (countOfActivation, countOfClasses) = len(self.layer[layerID].biases)
        
        outputVector = [None] * countOfClasses
        for cls in range(countOfClasses):
            for act in range(countOfActivation):
                outputVector[cls] += np.add(np.multiply(inputFeatureMap, self.layers[layerID].weights[0]), self.layers[layerID].biases[0]))
        
        #for backpropagation we need the X and W tensors
        cache = (inputFeatureMap, self.layers[layerID])
        
        return outputVector, cache
    

    #This function get the input and propagate the information forward to the output of CNN.
    #It is a pipeline. The output of each function is the input of the next one.
    #A for cycle iterate over all layers.
    def forward(self, X):
        cacheList = []
        #TODO reshape X if it is neccessary
        
        #first input
        #it called output because later we will call the functions with these name.
        output = X
        
        #For cycle to create the cnn pipeline.
        numberOfLAyers = len(self.layers)
        for layer in range(numberOfLAyers):
            if self.layers[layer].layerType == "convolutional":
                #TODO padding
                output, cache = self.convolution(output, layer)
                cacheList.append(cache)
                output = self.biasAdd(output)
                output = self.activationFunction(output, typeOfNonlinearity)    #NOTE: typeOfNonlinearity is not defined
                output = self.maxPooling2D(output, 2, 2, 2, 2)
            elif self.layers[layer].layerType == "fullyConnected":              #WARNING: fully connected before a convolutional layer is not working, check this.
                output, cache = self.convolution(fullyConnected, layer)
                
                output = self.activationFunction(output, typeOfNonlinearity)    #NOTE: typeOfNonlinearity is not defined
                
                
        return output, cacheList


    
    def backpropagation():
        
        
        return


    def train(self):
        #TODO train
        #TODO image load
        
        for epoch in range(numberOfEpochs):
                numberOfBatches = np.ceil(numberOfAllImages / batchSize)
                for batch in range(numberOfBatches):
                    #TODO loadImage + loadLabel
                    if batch == numberOfBatches:
                        batchSize = (numberOfAllImages - ((batch - 1) * batchSize))
                        batchX = imageLoader(sourceOfDB, batchSize)
                        batchY = labelLoader(sourceOfDB, batchSize)
                    else:
                        batchX = imageLoader(sourceOfDB, batchSize)
                        batchY = labelLoader(sourceOfDB, batchSize)
                        
                    
        
        

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
        
        
