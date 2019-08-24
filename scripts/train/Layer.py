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

		if(initializationType == "xavierUniform"):
			self.xavierUniform(weightSize)
		elif(initializationType == "xavierNormal"):
			self.xavierNormal(weightSize)
		elif(initializationType == "reluXavierUniform"):
			self.reluXavierUniform(weightSize)
		elif(initializationType == "reluXavierNormal"):
			self.reluXavierNormal(weightSize)
		elif(initializationType == "Orthogonal"):
			self.orthogonal(weightSize)
		else:
			raise Exception("Unhandled initialization type: " + str(initializationType))


	'''
	'' Determine the counts of input and output "neurons" (fans).
	'' These values will be the parameters of different types of initialization methods.
	'' Code source: https://github.com/keras-team/keras/blob/998efc04eefa0c14057c1fa87cab71df5b24bf7e/keras/initializations.py
	'' @param shape, is the layer structure, the convention is when the layer convulutional [N,H,W,C], [K, C] when is fully connected. Same with weightSize above.
	'''
	def getInputAndOutputFans(self, shape):		#TODO why it is called fans?
		#fanIn = shape[0] if len(shape) == 2 else np.prod(shape[1:])
		#fanOut = shape[1] if len(shape) == 2 else shape[0]
		if len(shape) == 2:
			#shape in this case is (N_in, N_out)
			fanIn = shape[0]	#in = N_in
			fanOut = shape[1]	#out = N_out
		else:
			#shape in this case is (N, H ,W, C)
			fanIn = numpy.prod(shape[1:])	#in = H*W*C
			fanOut = shape[0]		#out = N
		return fanIn, fanOut


	'''
	'' Get values from unirofm distribution.
	'' @param shape, is the layer structure, the convention is when the layer convulutional [N,H,W,C], [K, C] when is fully connected. Same with weightSize above.
	'' @param scale, values come from uniform distribution between -scale and +scale. 
	'''
	def weightsFromUniform(self, shape, scale=0.05):
		if len(shape) == 2:
			currentShape = shape
		else:
			currentShape = shape[1:]
			
		weights = numpy.random.uniform(low=-scale, high=scale, size=currentShape)
		
		return weights


	'''
	'' Get values from normal distribution.
	'' @param shape, is the layer structure, the convention is when the layer convulutional [N,H,W,C], [K, C] when is fully connected. Same with weightSize above.
	'' @param scale, values come from normal distribution with 0.0 mean and the deviation is scale.
	'''
	def weightsFromNormal(self, shape, scale=0.05):
		if len(shape) == 2:
			currentShape = shape
		else:
			currentShape = shape[1:]
			
		weights = numpy.random.normal(loc=0.0, scale=scale, size=currentShape)
		
		return weights


	'''
	'' Initialization of the weights and biases of the current layer.
	'' @param shape, is the layer structure, the convention is when the layer convulutional [N,H,W,C], [K, C] when is fully connected. Same with weightSize above.
	'' @param scale, values come from normal distribution with 0.0 mean and the deviation is scale.
	'''
	def doInitialization(self, shape, distributionType="uniformDistribution", distributionParameter=0.05):
		if(self.layerType == "Convolution"):
			numberOfKernels = shape[0]	#NOTE: it is same with fanOut
			if distributionType == "uniformDistribution":
				for kernelIdx in range(numberOfKernels):
					self.weights[kernelIdx] = self.weightsFromUniform(shape, distributionParameter)
					self.weights[kernelIdx] = self.weights[kernelIdx].astype(dtype=numpy.float128)
					self.biases[kernelIdx] = numpy.zeros(1)
					self.biases[kernelIdx] = self.biases[kernelIdx].astype(dtype=numpy.float128)
			elif distributionType == "normalDistribution":
				for kernelIdx in range(numberOfKernels):
					self.weights[kernelIdx] = self.weightsFromNormal(shape, distributionParameter)
					self.weights[kernelIdx] = self.weights[kernelIdx].astype(dtype=numpy.float128)
					self.biases[kernelIdx] = numpy.zeros(1)
					self.biases[kernelIdx] = self.biases[kernelIdx].astype(dtype=numpy.float128)
		elif(self.layerType == "FullyConnected"):
			numberOfOutPut = shape[1]
			if distributionType == "uniformDistribution":
				self.weights[0] = self.weightsFromUniform(shape, distributionParameter)
				self.weights[0] = self.weights[0].astype(dtype=numpy.float128)
				self.biases[0] = numpy.zeros((numberOfOutPut))
				self.biases[0] = self.biases[0].astype(dtype=numpy.float128)
			elif distributionType == "normalDistribution":
				self.weights[0] = self.weightsFromNormal(shape, distributionParameter)
				self.weights[0] = self.weights[0].astype(dtype=numpy.float128)
				self.biases[0] = numpy.zeros((numberOfOutPut))
				self.biases[0] = self.biases[0].astype(dtype=numpy.float128)
		else:
			raise Exception("Unhandled layer type: " + str(layerType))



	'''
	'' Initialize the weights based on Xavier method and biases with zeros.
	'' The principale came from this paper: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
	'' Reference: Glorot & Bengio, AISTATS 2010
	'' Note: The paper describe a similar method, but that distribution is uniform.
	'' @param shape, is the layer structure, the convention is when the layer convulutional [N,H,W,C], [K, C] when is fully connected.
	'''
	def xavierNormal(self, shape):
		#determine the input and output neurons
		fanIn, fanOut = self.getInputAndOutputFans(shape)
		
		#The parameter of the normal distribution.
		#The deviation means that how flat or sharp the distribution is.
		deviationOfDistribution = numpy.sqrt(2. / (fanIn + fanOut))
		
		self.doInitialization(shape, "normalDistribution", deviationOfDistribution)

	'''
	'' Initialize the weights based on Xavier method and biases with zeros.
	'' The principale came from this paper: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
	'' Reference: Glorot & Bengio, AISTATS 2010
	'' @param shape, is the layer structure, the convention is when the layer convulutional [N,H,W,C], [K, C] when is fully connected.
	'''
	def xavierUniform(self, shape):
		#determine the input and output neurons
		fanIn, fanOut = self.getInputAndOutputFans(shape)
		
		#The values come from between -limitOfDistribution and +limitOfDistribution.
		limitOfDistribution = numpy.sqrt(6. / (fanIn + fanOut))
		
		self.doInitialization(shape, "uniformDistribution", limitOfDistribution)


	'''
	'' Initialize the weights and biases.
	'' Weights come from normal distribution.
	'' The principale came from this paper: https://arxiv.org/pdf/1502.01852.pdf
	'' Reference:  He et al., http://arxiv.org/abs/1502.01852
	'' @param shape, is the layer structure, the convention is when the layer convulutional [N,H,W,C], [K, C] when is fully connected.
	'''
	def reluXavierNormal(self, shape):
		#determine the input and output neurons
		fanIn, fanOut = self.getInputAndOutputFans(shape)
		
		#The parameter of the normal distribution.
		#The deviation means that how flat or sharp the distribution is.
		deviationOfDistribution = numpy.sqrt(2. / fanIn)
		
		self.doInitialization(shape, "normalDistribution", deviationOfDistribution)

	'''
	'' Initialize the weights and biases.
	'' Weights come from normal distribution.
	'' The principale came from this paper: https://arxiv.org/pdf/1502.01852.pdf
	'' Reference:  He et al., http://arxiv.org/abs/1502.01852
	'' @param shape, is the layer structure, the convention is when the layer convulutional [N,H,W,C], [K, C] when is fully connected.
	'''
	def reluXavierUniform(self, shape):
		#determine the input and output neurons
		fanIn, fanOut = self.getInputAndOutputFans(shape)
		
		#The values come from between -limitOfDistribution and +limitOfDistribution.
		limitOfDistribution = numpy.sqrt(6. / fanIn)
		
		self.doInitialization(shape, "uniformDistribution", limitOfDistribution)

	'''
	'' Initialize the weights of layer with matrix orthogonalization.
	'' The principale came from this paper: From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
	'' NOTE: This is the only one initialization function, which is not use the doInitialization() function (TODO)
	'' @param shape, is the layer structure, the convention is when the layer convulutional [N,H,W,C], [K, C] when is fully connected.
	'''
	def orthogonal(self, shape):
		flatShape = (shape[0], numpy.prod(shape[1:]))
		
		#Generate random values (weights)
		randomWeights = numpy.random.normal(0.0, 1.0, flatShape)
		
		#Singular value decomposition for extract orthogonal matrices.
		#u is the left side eigenvectors, v is the right side eigenvectors and _ contains the singular values in diagonals.
		u, _, v = numpy.linalg.svd(randomWeights, full_matrices=False)
		
		# pick the one with the correct shape
		#q = u if u.shape == flatShape else v
		if u.shape == flatShape:
			q = u
		else:
			q = v
		
		#Reshape it back into original shape (because it flattened now).
		q = q.reshape(shape)
		
		#len(shape) == 2 means FullyConnected and else is the Convolutional layers.
		if len(shape) == 2:
			numberOfOutPut = shape[1]
			self.weights[0] = q
			self.weights[0] = self.weights[0].astype(dtype=numpy.float128)
			self.biases[0] = numpy.zeros((numberOfOutPut))
			self.biases[0] = self.biases[0].astype(dtype=numpy.float128)
		else:
			numberOfKernels = shape[0]	#NOTE: it is same with fanOut
			for kernelIdx in range(numberOfKernels):
				self.weights[kernelIdx] = q
				self.weights[kernelIdx] = self.weights[kernelIdx].astype(dtype=numpy.float128)
				self.biases[kernelIdx] = numpy.zeros(1)
				self.biases[kernelIdx] = self.biases[kernelIdx].astype(dtype=numpy.float128)