'''
root/codes/block_definitions/utilities/operators...

Overview:
Here we are relying on the Augmentor module to handle the 'pipeline' of our data or image processing.
Augmentor has a super simple way to do batching, and on top of that add in data preprocessing and augmentation.
Here we assume that the input and output is an Augmentor.Pipeline object and each primitive changes or adds an attribute
to that object which will eventually change how data is read in to our neural network.
Here we will be adding transfer learning with precomputed neural networks; we will add them to the pipeline as if it is a 
preprocessing operation we are adding...
https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.add_operation
https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Operations.Operation

Rules:
Since we are manipulating an object instead of actually evaluating something, I decided to deepcopy the object
before returning it. In hindsight this shouldn't make a difference to the final result but it could make debugging
easier if the output from each node ends up being different things instead of different variables pointing to the same object.
'''

### packages
import Augmentor
from Augmentor.Operations import Operation
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet50 import preprocess_input
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



class ResNet(Operation):
    '''    
    Purpose of function is to add an Augmentor primtive that is literally the output of a fully trained neural network
    '''
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability=1, resModel=None, cutModel=None):
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability=probability)
        self.resModel = resModel
        self.cutModel = cutModel

    # Your class must implement the perform_operation method:
    def perform_operation(self, images):
        # Return the image so that it can further processed in the pipeline:
        images = np.asarray(images)
        resModel = ResNet152V2(include_top=False,
                               weights='imagenet',
                               input_tensor=None,
                               input_shape=images[0].shape)
        out = resModel.predict(images)
        tf.keras.backend.clear_session()
        return out


"""https://keras.io/applications/"""
def res_net(pipeline):
    pipeline.add_operation(ResNet())
    return deepcopy(pipeline)

operator_dict[res_net] = {"inputs": [Augmentor.Pipeline],
                          "output": Augmentor.Pipeline,
                          "args": []
                         }



class ResNetNorm(Operation):
    '''    
    Purpose of function is to add an Augmentor primtive that is literally the output of a fully trained neural network
    '''
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability=1):
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability=probability)

    # Your class must implement the perform_operation method:
    def perform_operation(self, images):
        # Return the image so that it can further processed in the pipeline:
        images = np.asarray(images)
        out =  preprocess_input(images) #TODO: talk to Jurado about this...is there a better way? should we call in Resnet50 like we do above with ResNet15V2?
        tf.keras.backend.clear_session()
        return out


def res_net_norm(pipeline):
    pipeline.add_operation(ResNetNorm())
    return deepcopy(pipeline)

operator_dict[res_net_norm] = {"inputs": [Augmentor.Pipeline],
                               "output": Augmentor.Pipeline,
                               "args": []
                              }