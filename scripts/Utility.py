#!/usr/bin/env python

# Scientific computing library.
import numpy

'''
'' Stores the static methods which used in the CNN.
''
'' TODO This class need to move into another script (Utility.py)
''
'' @author Nagy Marton
'' @author Kisházi "janohhank" János
'''
class Utility:
	'''
	'' Returns the selected activation function.
	'' @param activationFunctionType, is the name of the activation function.
	'''
	@staticmethod
	def getActivationFunction(self, activationFunctionType):
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
	def getDerivitiveActivationFunction(self, activationFunctionType):
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
			softmax = Utility.getActivationFunction('softmax')
			return lambda x :softmax(x)*(1-softmax(x))
		else:
			raise Exception("Unhandled activation function type: " + str(activationFunctionType))