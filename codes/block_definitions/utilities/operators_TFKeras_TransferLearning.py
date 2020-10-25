'''
root/codes/block_definition/utilities/operators...

Overview:
New approach to transfer learning (replaces augmentor)

Rules:
Here the operator/primitive input args are super unique to the rest of ezcgp.
The assumption is that we will never evaluate on more than one node in a genome, and that
all we are doing when 'evaluating' is starting our tf.keras.Model.
So we don't pass anything into the operator except the expected input image shape.
This sort of makes the operator_dict useless especially for the input and output keys. But
we keep it in and set them to tf.keras.layes just for consistency.
All operators should return the first and last layer of the pretrained network that we want 
to utilize.
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


def vgg16(input_shape):
    '''
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16

    Note on input image shape: "It should have exactly 3 input channels,
    and width and height should be no smaller than 32."
    '''
    pretrained_model = tf.keras.applications.VGG16(include_top=False,
                                                   weights='imagenet',
                                                   input_tensor=None,
                                                   input_shape=input_shape,
                                                   pooling=None)
    return pretrained_model.inputs, pretrained_model.outputs

operator_dict[vgg16] = {"inputs": [tf.keras.layers],
                        "output": tf.keras.layers,
                        "args": []
                        }


def resnet(input_shape, ith_model):
    '''
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2
    
    There are 3 architectures listed:
     * ResNet101V2
     * ResNet152V2
     * ResNet50V2

    Note on input image shape: "It should have exactly 3 input channels,
    and width and height should be no smaller than 32."
    '''
    models = [tf.keras.applications.ResNet101V2, tf.keras.applications.ResNet152V2, tf.keras.applications.ResNet50V2]
    ith_model = ith_model % len(models)
    pretrained_model = tf.keras.applications.ResNet152V2(include_top=False,
                                                         weights='imagenet',
                                                         input_tensor=None,
                                                         input_shape=input_shape,
                                                         pooling=None)
    return pretrained_model.inputs, pretrained_model.outputs

operator_dict[resnet] = {"inputs": [tf.keras.layers],
                        "output": tf.keras.layers,
                        "args": [argument_types.ArgumentType_Int0to25]
                        }


