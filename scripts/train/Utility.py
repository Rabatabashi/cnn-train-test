#!/usr/bin/env python

# Scientific computing library.
import numpy

'''
'' Stores the static methods which used in the CNN.
''
'' @author Nagy "rabatabashi" Marton
'' @author Kishazi "janohhank" Janos
'''
class Utility:
	'''
	'' Returns the selected activation function.
	'' @param activationFunctionType, is the name of the activation function.
	'''
	@staticmethod
	def getActivationFunction(activationFunctionType):
		if(activationFunctionType == "sigmoid"):
			return lambda x : numpy.exp(x)/(1 + numpy.exp(x))
		elif(activationFunctionType == "linear"):
			return lambda x : x
		elif(activationFunctionType == "relu"):
			def relu(x):
				y = numpy.copy(x)
				y[y<0] = 0
				return y
			return relu
		elif(activationFunctionType == "softmax"):
			def softmax(x):
				xx = numpy.copy(x)
				y = numpy.exp(xx) / numpy.sum(numpy.exp(xx))
				return y
			return softmax
		else:
			raise Exception("Unhandled activation function type: " + str(activationFunctionType))

	'''
	'' Returns the selected activation function derivate.
	'' @param activationFunctionType, is the name of the activation function.
	'''
	@staticmethod
	def getDerivitiveActivationFunction(activationFunctionType):
		if(activationFunctionType == "sigmoid"):
			sig = lambda x : numpy.exp(x)/(1 + numpy.exp(x))
			return lambda x :sig(x)*(1-sig(x))
		elif(activationFunctionType == "linear"):
			return lambda x: 1
		elif(activationFunctionType == "relu"):
			def relu_diff(x):
				y = numpy.copy(x)
				y[y>=0] = 1
				y[y<0] = 0
				return y
			return relu_diff
		elif(activationFunctionType == "softmax"):
			def gradSoftMax(x):
				softmax = Utility.getActivationFunction('softmax')

				y = numpy.copy(x)
				length = len(y)

				gradSM = numpy.zeros((length, length))

				for i in range(length):
					for j in range(length):
						if i == j:
							gradSM[i, j] = softmax(y[i])*(1-softmax(y[j]))
						else:
							gradSM[i, j] = softmax(y[i])*(-softmax(y[j]))
				return gradSM
			return gradSoftMax
		else:
			raise Exception("Unhandled activation function type: " + str(activationFunctionType))