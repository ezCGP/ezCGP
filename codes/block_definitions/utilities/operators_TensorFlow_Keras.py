'''
root/codes/block_definition/utilities/operators...

Overview:
going to add basically any method listed in this module
www.tensorflow.org/api_docs/python/tf/keras/layers

Rules:
Make sure the BlockEvaluate Definition is appropriate to the tf.keras module.
And make sure argument types are valid and appropriate for your needs
'''

### packages
import tensorflow as tf
import numpy as np

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### abosulte imports wrt root
from codes.utilities.custom_logging import ezLogging
from codes.block_definitions.utilities import argument_types

# init dict
operator_dict = {}


def conv2D_layer(input_tensor, filters=64, kernel_size=3, activation=tf.nn.relu):
    kernel_size = (kernel_size, kernel_size)
    '''
    Convolutional Layer
    Computes 32 features using a 5x5 filter with ReLU activation.
    Padding is added to preserve width and height.

    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
    '''
    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding="same",
                                  activation=activation,
                                  data_format="channels_last"
                                 )(input_tensor)

operator_dict[conv2D_layer] = {"inputs": [tf.keras.layers.Layer],
                               "output": tf.keras.layers.Layer,
                               "args": [argument_types.ArgumentType_Pow2,
                                        argument_types.ArgumentType_TFFilterSize,
                                        argument_types.ArgumentType_TFActivation]
                              }


def conv2DTranspose_layer(input_tensor, filters=64, kernel_size=3, activation=tf.nn.relu):
    kernel_size = (kernel_size, kernel_size)
    '''
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose
    '''
    return tf.keras.layers.Conv2DTranspose(filters=filters,
                                           kernel_size=kernel_size,
                                           padding="same",
                                           activation=activation,
                                           data_format="channels_last"
                                           )(input_tensor)
'''
operator_dict[conv2DTranspose_layer] = {"inputs": [tf.keras.layers.Layer],
                                        "output": tf.keras.layers.Layer,
                                        "args": [argument_types.ArgumentType_Pow2,
                                                 argument_types.ArgumentType_TFFilterSize,
                                                 argument_types.ArgumentType_TFActivation]
                                       }
'''

''' not useful for images
def conv3D_layer(input_tensor, filters=64, kernel_size=3, activation=tf.nn.relu):
    #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D
    kernel_size = (kernel_size, kernel_size, kernel_size)
    return tf.keras.layers.Conv3D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding="same",
                                  activation=activation,
                                  data_format="channels_last"
                                 )(input_tensor)

operator_dict[conv3D_layer] = {"inputs": [tf.keras.layers.Layer],
                               "output": tf.keras.layers.Layer,
                               "args": [argument_types.ArgumentType_Pow2,
                                        argument_types.ArgumentType_TFFilterSize,
                                        argument_types.ArgumentType_TFActivation]
                              }'''


''' not useful for images
def conv3DTranspose_layer(input_tensor, filters=64, kernel_size=3, activation=tf.nn.relu):
    #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3DTranspose
    kernel_size = (kernel_size, kernel_size, kernel_size)
    return tf.keras.layers.Conv3DTranspose(filters=filters,
                                           kernel_size=kernel_size,
                                           padding="same",
                                           activation=activation,
                                           data_format="channels_last"
                                          )(input_tensor)

operator_dict[conv3DTranspose_layer] = {"inputs": [tf.keras.layers.Layer],
                                        "output": tf.keras.layers.Layer,
                                        "args": [argument_types.ArgumentType_Pow2,
                                                 argument_types.ArgumentType_TFFilterSize,
                                                 argument_types.ArgumentType_TFActivation]
                                       }'''


def gaussianNoise_layer(input_tensor, stddev):
    '''
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/GaussianNoise
    '''
    return tf.keras.layers.GaussianNoise(stddev=stddev)(input_tensor)

# operator_dict[gaussianNoise_layer] = {"inputs": [tf.keras.layers.Layer],
#                                         "output": tf.keras.layers.Layer,
#                                         "args": [argument_types.ArgumentType_Float0to1]
#                                        }


### Pooling Layers

def avg_pool_layer(input_tensor, pool_size=2):
    return tf.keras.layers.AveragePooling2D(pool_size=[pool_size, pool_size], padding="valid")(input_tensor)

operator_dict[avg_pool_layer] = {"inputs": [tf.keras.layers.Layer],
                                 "output": tf.keras.layers.Layer,
                                 "args": [argument_types.ArgumentType_TFPoolSize]
                                 }


def max_pool_layer(input_tensor, pool_size=2):
    return tf.keras.layers.MaxPool2D(pool_size=[pool_size, pool_size], padding="valid")(input_tensor)

operator_dict[max_pool_layer] = {"inputs": [tf.keras.layers.Layer],
                                 "output": tf.keras.layers.Layer,
                                 "args": [argument_types.ArgumentType_TFPoolSize]
                                 }


def fractional_max_pool(input_tensor, pool_height=2, pool_width=2):
    if input_tensor.shape[1] == 1:
        return input_tensor
    # see args.py for mutation limits
    pooling_ratio = [1.0, pool_height, pool_width, 1.0]
    # true random underfits when combined with data augmentation and/or dropout
    pseudo_random = True
    # overlapping pooling regions work better, according to 2015 Ben Graham paper
    overlapping = True
    # returns a tuple of Tensor objects (output, row_pooling_sequence, col_pooling_sequence
    return tf.nn.fractional_max_pool(input_tensor, pooling_ratio, pseudo_random, overlapping)[0]

# operator_dict[fractional_max_pool] = {"inputs": [tf.keras.layers.Layer],
#                                       "args": [argument_types.ArgumentType_TFPoolSize, argument_types.ArgumentType_TFPoolSize],
#                                       "output": tf.keras.layers.Layer
#                                       }


def fractional_avg_pool(input_tensor, pool_height=2.0, pool_width=2.0):
    if input_tensor.shape[1].value == 1:
        return input_tensor
    pooling_ratio = [1.0, pool_height, pool_width, 1.0]
    pseudo_random = True
    overlapping = True
    # returns a tuple of Tensor objects (output, row_pooling_sequence, col_pooling_sequence)
    return tf.nn.fractional_avg_pool(input_tensor, pooling_ratio, pseudo_random, overlapping)[0]

# operator_dict[fractional_avg_pool] = {"inputs": [tf.keras.layers.Layer],
#                                       "args": [argument_types.ArgumentType_TFPoolSize, argument_types.ArgumentType_TFPoolSize],
#                                       "output": tf.keras.layers.Layer
#                                       }


def dropout_layer(input_tensor, rate=0.2):
    return tf.keras.layers.Dropout(rate/2)(input_tensor)

operator_dict[dropout_layer] = {"inputs": [tf.keras.layers.Layer],
                                "output": tf.keras.layers.Layer,
                                "args": [argument_types.ArgumentType_Float0to1]
                                }
