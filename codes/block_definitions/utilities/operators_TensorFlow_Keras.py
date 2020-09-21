'''
root/codes/block_definition/utilities/operators...

Overview:
going to add basically any method listed in this module
www.tensorflow.org/api_docs/python/tf/keras/layers

Rules:
Make sure the BlockEvaluate Definition is appropriate to the tf.keras module.
And make sure argument types are valid and appropriate for your needs
'''


# packages
import tensorflow as tf
import numpy as np

# sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

# abosulte imports wrt root
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


operator_dict[conv2D_layer] = {"inputs": [tf.keras.layers],
                               "output": tf.keras.layers,
                               "args": [argument_types.ArgumentType_Pow2, argument_types.ArgumentType_TFFilterSize, argument_types.ArgumentType_TFActivation]
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


operator_dict[conv2DTranspose_layer] = {"inputs": [tf.keras.layers],
                                        "output": tf.keras.layers,
                                        "args": [argument_types.ArgumentType_Pow2, argument_types.ArgumentType_TFFilterSize, argument_types.ArgumentType_TFActivation]
                                        }

def conv3D_layer(input_tensor, filters=64, kernel_size=3, activation=tf.nn.relu):
    kernel_size = (kernel_size, kernel_size)
    '''
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D
    '''

    return tf.keras.layers.Conv3D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding="same",
                                  activation=activation,
                                  data_format="channels_last"
                                  )(input_tensor)


operator_dict[conv3D_layer] = {"inputs": [tf.keras.layers],
                               "output": tf.keras.layers,
                               "args": [argument_types.ArgumentType_Pow2, argument_types.ArgumentType_TFFilterSize, argument_types.ArgumentType_TFActivation]
                               }


def conv3DTranspose_layer(input_tensor, filters=64, kernel_size=3, activation=tf.nn.relu):
    kernel_size = (kernel_size, kernel_size)
    '''
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3DTranspose
    '''

    return tf.keras.layers.Conv3DTranspose(filters=filters,
                                           kernel_size=kernel_size,
                                           padding="same",
                                           activation=activation,
                                           data_format="channels_last"
                                           )(input_tensor)


operator_dict[conv3DTranspose_layer] = {"inputs": [tf.keras.layers],
                                        "output": tf.keras.layers,
                                        "args": [argument_types.ArgumentType_Pow2, argument_types.ArgumentType_TFFilterSize, argument_types.ArgumentType_TFActivation]
                                        }