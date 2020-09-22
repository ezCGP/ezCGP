'''
root/codes/block_definitions/utilities/operators...

Overview:
Here we are relying on the Augmentor module to handle the 'pipeline' of our data or image processing.
Augmentor has a super simple way to do batching, and on top of that add in data preprocessing and augmentation.
Here we assume that the input and output is an Augmentor.Pipeline object and each primitive changes or adds an attribute
to that object which will eventually change how data is read in to our neural network.
All of these primitives will be Augmentor.Operations classes and we will use Augmentor.Pipeline.add_operation(method)
to add them
https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.add_operation
https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Operations.Operation

A lot of the methods are going to be 'transfer learning' type models from 
https://www.tensorflow.org/api_docs/python/tf/keras/applications

Rules:
Since we are manipulating an object instead of actually evaluating something, I decided to deepcopy the object
before returning it. In hindsight this shouldn't make a difference to the final result but it could make debugging
easier if the output from each node ends up being different things instead of different variables pointing to the same object.
'''

### packages
import Augmentor
import numpy as np
from copy import deepcopy
from tensorflow.keras.applications import resnet_v2, ResNet101V2, ResNet152V2, ResNet50V2
from tensorflow.keras.applications import vgg16, VGG16
from tensorflow.keras.applications import inception_v3, InceptionV3


### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.block_definitions.utilities import argument_types
from codes.utilities.custom_logging import ezLogging


### init dict
operator_dict = {}



class ResNet(Augmentor.Operations.Operation):
    '''
    a little bit of transfer learning!
    
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2
    
    There are 3 architectures listed:
     * ResNet101V2
     * ResNet152V2
     * ResNet50V2
    '''
    def __init__(self, ith_model=0, probability=1):
        self.__init__(probability=probability)
        models = [ResNet101V2, ResNet152V2, ResNet50V2]
        ith_model = ith_model % len(models)
        self.resnet = models[ith_model]
    
    def perform_operation(self, images):
        '''
        assuming the images are already between [0,255]
        TF Keras documentation says
            "Caution: Be sure to properly pre-process your inputs to the application."
        https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/preprocess_input
        
        ...which is why we call the preprocessing
        '''
        images = resnet_v2.preprocess_input(np.asarray(images))
        model = self.resnet(include_top=False,
                            weights='imagenet',
                            input_tensor=None,
                            input_shape=images[0].shape,
                            pooling=None,
                            classes=None, # ignored when include_top False
                            classifier_activation=None, # ignored when include_top False
                           )
        feature_maps = model.predict(images)
        tf.keras.backend.clear_session()
        return feature_maps


def resnet(pipeline, ith_model, probability=1):
    pipeline.add_operation(ResNetv2(ith_model, probability))
    return deepcopy(pipeline)


operator_dict[resnet] = {"inputs": [Augmentor.Pipeline],
                         "output": Augmentor.Pipeline,
                         "args": [ArgumentType_Int0to25]
                        }



class VGG(Augmentor.Operations.Operation):
    '''
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16
    '''
    def __init__(self, probability=1):
        super().__init__(probability=1)

    def perform_operation(self, images):
        images = vgg16.preprocess_input(np.asarray(images))
        model = VGG16(include_top=False,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=images[0].shape,
                      pooling=None,
                      classes=None, # ignored when include_top False
                      classifier_activation=None, # ignored when include_top False
                     )
        feature_maps = model.predict(images)
        tf.keras.backend.clear_session()
        return feature_maps


def vgg(pipeline, probability=1):
    pipeline.add_operation(VGG(probability))
    return deepcopy(pipeline)


operDict[vgg] = {"inputs": [Augmentor.Pipeline],
                 "output": Augmentor.Pipeline,
                 "args": []
                }



class Inception(Augmentor.Operations.Operation):
    '''
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3
    '''
    def __init__(self, probability=1):
        super().__init__(probability=1)

    def perform_operation(self, images):
        images = inception_v3.preprocess_input(np.asarray(images))
        model = InceptionV3(include_top=False,
                            weights='imagenet',
                            input_tensor=None,
                            input_shape=images[0].shape,
                            pooling=None,
                            classes=None, # ignored when include_top False
                            classifier_activation=None, # ignored when include_top False
                           )
        feature_maps = model.predict(images)
        tf.keras.backend.clear_session()
        return feature_maps


def inception(pipeline, probability=1):
    pipeline.add_operation(Inception(probability))
    return deepcopy(pipeline)


operDict[inception] = {"inputs": [Augmentor.Pipeline],
                       "output": Augmentor.Pipeline,
                       "args": []
                      }

