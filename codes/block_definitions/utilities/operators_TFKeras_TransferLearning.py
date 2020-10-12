'''
root/codes/block_definition/utilities/operators...

Overview:
New approach to transfer learning (replaces augmentor)

Rules:
'''
### packages
import tensorflow as tf
import numpy as np

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### abosulte imports wrt root
from codes.block_definitions.utilities import argument_types

# init dict
operator_dict = {}


def vgg16(input_layers):
    '''
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16
    '''
    pretrained_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling=None)
    return tf.keras.layers.Add()[input_layers , pretrained_model.layers[-1].output]

operator_dict[vgg16] = {"inputs": [tf.keras.layers],
                        "output": tf.keras.layers,
                        "args": []
                        }


def resnet(input_layers):
    '''
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet152V2
    '''
    pretrained_model = tf.keras.applications.ResNet152V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling=None)
    return tf.keras.layers.Add()[input_layers , pretrained_model.layers[-1].output]

operator_dict[resnet] = {"inputs": [tf.keras.layers],
                        "output": tf.keras.layers,
                        "args": []
                        }


