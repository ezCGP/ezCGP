'''
root/codes/block_definitions/utilities/operators...

Overview:
NOT OPERATIONAL

Rules:
'''

### packages
import tensorflow as tf
import numpy as np
from copy import deepcopy

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.block_definitions.utilities import argument_types
from codes.utilities.custom_logging import ezLogging


### init dict
operator_dict = {}


def relu_func(input):
    # ReLu Non-linear activation function
    return tf.keras.layers.ReLu()(input)

def sigmoid_func(input):
    return tf.keras.activations.sigmoid()(input)


def activation(input):
    return tf.keras.layers.ReLU()(input)
