#!/usr/bin/env python

# Scientific computing library.
import numpy

'''
'' Neural network layer abstraction class which can be a convolutional layer or a fully connected layer.
'' Initialize the weights of the kernels and the bias vectors elements.
''
'' @author Nagy Marton
'' @author Kishazi "janohhank" Janos
'''
class Layer:
	# The layer sequence number of neural network.
	layerID = None

	# The layer type, it can be Convolution or FullyConnected.
	layerType = None

	# The layer activation (nonlinearity) function type.
	activationType = None

	# Stored kernels weights map.
	weights = None

	# Stored bias vectors map.
	biases = None

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
		self.weights = {}
		self.biases = {}

		if(initializationType == "UniformRandom"):
			self.simpleUniformInitialization(weightSize)
		elif(initializationType == "Xavier"):
			self.xavierInitialization(weightSize)
		else:
			raise Exception("Unhandled initialization type: " + str(initializationType))

	'''
	'' TODO DOC
	'''
	def simpleUniformInitialization(self, weightSize):
		if(self.layerType == "Convolution"):
			numberOfKernels = weightSize[0]
			kernelHeight = weightSize[1]
			kernelWidth = weightSize[2]
			channels = weightSize[3]

			for kernelIdx in range(numberOfKernels):
				self.weights[kernelIdx] = numpy.random.uniform(-0.01, +0.01, (kernelHeight, kernelWidth, channels))
				self.weights[kernelIdx] = self.weights[kernelIdx].astype(dtype=numpy.float128)
				self.biases[kernelIdx] = numpy.random.uniform(-0.01, +0.01, (1))
				self.biases[kernelIdx] = self.biases[kernelIdx].astype(dtype=numpy.float128)
		elif(self.layerType == "FullyConnected"):
			# TODO it must calculate instead of arbitrary give it.
			lastFeatureMapSize = weightSize[0]
			numberOfOutPut = weightSize[1]

			self.weights[0] = numpy.random.uniform(-0.01, +0.01, (lastFeatureMapSize, numberOfOutPut))
			self.weights[0] = self.weights[0].astype(dtype=numpy.float128)
			self.biases[0] = numpy.random.uniform(-0.01, +0.01, (numberOfOutPut))
			self.biases[0] = self.biases[0].astype(dtype=numpy.float128)
		else:
			raise Exception("Unhandled layer type: " + str(layerType))

	'''
	'' TODO DOC
	'''
	def xavierInitialization(self, weightSize):
		