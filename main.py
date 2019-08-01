#!/usr/bin/python3
#
# This python script train a fully connected neural network with multiple layers.
#
# author Nagy Marton

import argparse
import numpy as np


#ARGPARSER
parser = argparse.ArgumentParser(description='Process some integers.')

#parser.add_argument("--inputParametersFile", required=True,  type=int, help="The path of file which includes the parameters of CNN and hyper-parameters of train.")
parser.add_argument("--inputSize", required=True,  type=int, help="The dimension of input images. This will be the height and width of input of CNN.")  #NOTE: 28px x 28px at MNIST
parser.add_argument("--countOfClasses", required=True,  type=int, help="Number of classes.")  #NOTE: 10 class at MNIST
args = parser.parse_args()

inputSize = args.inputSize
countOfClasses = args.countOfClasses


weightSizes = [
    [5, 3, 3, 1],
    [10, 3, 3, 5],
    [(inputSize/(2**2))*(inputSize/(2**2)) * 10, countOfClasses]
]

print(weightSizes)