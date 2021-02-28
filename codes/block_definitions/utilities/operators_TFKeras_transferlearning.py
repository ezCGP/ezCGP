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



class InputImageTooSmall(Exception):
    '''
    customized exception from this example:
    https://www.programiz.com/python-programming/user-defined-exception

    Exception raised if the pretrained model has a minimum image height x width
    dimensions not met by the image batch.

    Attributes:
        image_dimension -- the invalid value of the image
        minimum_size -- the smallest valid dimension as defined by the tf.keras doc
    '''
    def __init__(self, image_dimension, minimum_size):
        self.image_dimension = image_dimension
        self.minimum_size = minimum_size
        self.message = "Input image is smaller than minimum (%i,%i)" % (minimum_size,minimum_size)
        super().__init__(self.message)


    def __str__(self):
        return f'{self.image_dimension} -> {self.message}'


def check_size_compatibility(input_layer, minimum_size):
    batch_size, height, width, channels = input_layer.shape # batch_size prob is None
    if (height < minimum_size) or (width < minimum_size):
        smallest_dim = min(height, width)
        raise InputImageTooSmall(smallest_dim, minimum_size)



def vgg16(input_layer):
    '''
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16

    Note on input image shape: "It should have exactly 3 input channels,
    and width and height should be no smaller than 32."
    '''
    check_size_compatibility(input_layer, 32)
    next_layer = tf.keras.applications.vgg16.preprocess_input(input_layer)
    pretrained_model = tf.keras.applications.VGG16(include_top=False,
                                                   weights='None',
                                                   input_tensor=None,
                                                   input_shape=input_layer.shape[1:], #ignore 0th element batch_size
                                                   pooling=None)
    output_layer = pretrained_model(next_layer)
    return output_layer

operator_dict[vgg16] = {"inputs": [tf.keras.layers],
                        "output": tf.keras.layers,
                        "args": []
                        }


def resnet(input_layer, ith_model):
    '''
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2
    
    There are 3 architectures listed:
     * ResNet101V2
     * ResNet152V2
     * ResNet50V2

    Note on input image shape: "It should have exactly 3 input channels,
    and width and height should be no smaller than 32."
    '''
    check_size_compatibility(input_layer, 32)
    next_layer = tf.keras.applications.resnet_v2.preprocess_input(input_layer)
    models = [tf.keras.applications.ResNet101V2, tf.keras.applications.ResNet152V2, tf.keras.applications.ResNet50V2]
    ith_model = ith_model % len(models)
    pretrained_model = models[ith_model](include_top=False,
                                         weights='None',
                                         input_tensor=None,
                                         input_shape=input_layer.shape[1:], #ignore 0th element batch_size
                                         pooling=None
                                        )
    output_layer = pretrained_model(next_layer)
    return output_layer

operator_dict[resnet] = {"inputs": [tf.keras.layers],
                        "output": tf.keras.layers,
                        "args": [argument_types.ArgumentType_Int0to25]
                        }


def inception(input_layer):
    '''
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3

    Note limit on input image shape: "It should have exactly 3 inputs channels,
    and width and height should be no smaller than 75"...going to kill otherwise
    '''
    check_size_compatibility(input_layer, 75)
    next_layer = tf.keras.applications.inception_v3.preprocess_input(input_layer)
    pretrained_model = tf.keras.applications.InceptionV3(include_top=False,
                                                         weights='None',
                                                         input_tensor=None,
                                                         input_shape=input_layer.shape[1:], #ignore 0th element batch_size
                                                         pooling=None)
    output_layer = pretrained_model(next_layer)
    return output_layer


operator_dict[inception] = {"inputs": [tf.keras.layers],
                            "output": tf.keras.layers,
                            "args": []
                           }

