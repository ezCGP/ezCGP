'''
root/codes/block_definitions/utilities/operators...

Overview:
strictly just tensorflow pooling layers

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


def avg_pool_layer(input_tensor, pool_height=2, pool_width=2, strides=2):
    #return tf.layers.average_pooling2d(inputs=input_tensor, pool_size=[2,2], strides=2)
    return tf.keras.layers.MaxPool2D(pool_size=[pool_height, pool_width], strides=strides, padding="valid")(input_tensor)

operator_dict[avg_pool_layer] = {"inputs": [tf.keras.layers],
                                 "output": tf.keras.layers,
                                 "args": [argument_types.ArgumentType_TFPoolSize,
                                          argument_types.ArgumentType_TFPoolSize,
                                          argument_types.ArgumentType_TFPoolSize] # TODO verify the argtype we want for strieds...i just picked one
                                }


def max_pool_layer(input_tensor, pool_height=2, pool_width=2, strides=2):
    if input_tensor.shape[1].value == 1:
        return input_tensor
    else:
        return tf.keras.layers.MaxPool2D(pool_size=[pool_height, pool_width], strides=strides, padding="valid")(input_tensor)

operator_dict[max_pool_layer] = {"inputs": [tf.keras.layers],
                                 "output": tf.keras.layers,
                                 "args": [argument_types.ArgumentType_TFPoolSize,
                                          argument_types.ArgumentType_TFPoolSize,
                                          argument_types.ArgumentType_TFPoolSize]
                                }


def fractional_max_pool(input_tensor, pool_height=2, pool_width=2):
    if input_tensor.shape[1] == 1:
        return input_tensor
    pooling_ratio = [1.0, pool_height, pool_width, 1.0]  # see args.py for mutation limits
    pseudo_random = True  # true random underfits when combined with data augmentation and/or dropout
    overlapping = True  # overlapping pooling regions work better, according to 2015 Ben Graham paper
    # returns a tuple of Tensor objects (output, row_pooling_sequence, col_pooling_sequence
    return tf.nn.fractional_max_pool(input_tensor, pooling_ratio, pseudo_random, overlapping)[0]

operator_dict[fractional_max_pool] = {"inputs": [tf.keras.layers],
                                      "args": [argument_types.ArgumentType_TFPoolSize, argument_types.ArgumentType_TFPoolSize],
                                      "output": tf.keras.layers
                                     }


def fractional_avg_pool(input_tensor, pool_height=2.0, pool_width=2.0):
    if input_tensor.shape[1].value == 1:
        return input_tensor
    pooling_ratio = [1.0, pool_height, pool_width, 1.0]
    pseudo_random = True
    overlapping = True
    # returns a tuple of Tensor objects (output, row_pooling_sequence, col_pooling_sequence)
    return tf.nn.fractional_avg_pool(input_tensor, pooling_ratio, pseudo_random, overlapping)[0]

operator_dict[fractional_avg_pool] = {"inputs": [tf.keras.layers],
                                      "args": [argument_types.ArgumentType_TFPoolSize, argument_types.ArgumentType_TFPoolSize],
                                      "output": tf.keras.layers
                                     }