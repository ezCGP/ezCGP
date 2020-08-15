'''
root/codes/block_definitions/utilities/operators...

Overview:
strictly just tensorflow layer operators

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


def sub_tensors(input_layer0, input_layer1):
    return tf.keras.layers.Subtract([input_layer0, input_layer1])

operator_dict[sub_tensors] = {"inputs": [tf.keras.layers, tf.keras.layers],
                              "output": tf.keras.layers,
                              "args": []
                             }


def mult_tensors(input_layer0, input_layer1):
    return keras.layers.Multiply([input_layer0, input_layer1])

operator_dict[mult_tensors] = {"inputs": [tf.keras.layers, tf.keras.layers],
                               "output": tf.keras.layers,
                               "args": []
                              }


def add_tensors(input_layer0, input_layer1):
    return tf.keras.layers.Add()([input_layer0, input_layer1])

operator_dict[add_tensors] = {"inputs": [tf.keras.layers, tf.keras.layers],
                              "output": tf.keras.layers,
                              "args": []
                             }


def flatten_layer(input_layer):
    return tf.keras.layers.Flatten()(input_layer)

operator_dict[flatten_layer] = {"inputs": [tf.keras.layers],
                                "output": tf.keras.layers,
                                "args":[]
                               }


def dropout_layer(input_layer):
    return tf.keras.layers.Dropout(rate=0.2)(input_layer)

operator_dict[dropout_layer] = {"inputs": [tf.keras.layers],
                                "output": tf.keras.layers,
                                "args": []
                               }