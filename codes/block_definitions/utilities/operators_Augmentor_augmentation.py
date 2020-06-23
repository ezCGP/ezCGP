'''
root/codes/block_definitions/utilities/operators...

Overview:
Here we are relying on the Augmentor module to handle the 'pipeline' of our data or image processing.
Augmentor has a super simple way to do batching, and on top of that add in data preprocessing and augmentation.
Here we assume that the input and output is an Augmentor.Pipeline object and each primitive changes or adds an attribute
to that object which will eventually change how data is read in to our neural network.

Rules:
Since we are manipulating an object instead of actually evaluating something, I decided to deepcopy the object
before returning it. In hindsight this shouldn't make a difference to the final result but it could make debugging
easier if the output from each node ends up being different things instead of different variables pointing to the same object.
'''

### packages
import Augmentor
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


def rotate(pipeline, probability=.5, max_rotation=10):
    '''
    https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.rotate

    prob: float (0,1]
    max_left/right_rotation: int [0,25]

    going to assign max_rotation to both left and right
    '''
    pipeline.rotate(probability=probability, max_left_rotation=max_rotation, max_right_rotation=max_rotation)
    return deepcopy(pipeline)

operator_dict[rotate] = {"inputs": [Augmentor.Pipeline],
                         "output": Augmentor.Pipeline,
                         "args": [argument_types.ArgumentType_LimitedFloat0to1, argument_types.ArgumentType_Int0to25]
                        }


def horizontal_flip(pipeline, probability=.5):
    '''
    https://arxiv.org/pdf/1912.11370v2.pdf
    https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.flip_left_right
    prob: float (0,1]
    '''
    pipeline.flip_left_right(probability)
    return deepcopy(pipeline)

operator_dict[horizontal_flip] = {"inputs": [Augmentor.Pipeline],
                                  "output": Augmentor.Pipeline,
                                  "args": [argument_types.ArgumentType_LimitedFloat0to1]
                                 }


def random_crop(pipeline, probability=.2):
    '''
    https://arxiv.org/pdf/1912.11370v2.pdf

    '''
    pipeline.flip_left_right(probability) #should be different!
    return deepcopy(pipeline)

operator_dict[random_crop] = {"inputs": [Augmentor.Pipeline],
                              "output": Augmentor.Pipeline,
                              "args": [argument_types.ArgumentType_LimitedFloat0to1]  # this will choose values between 0 and 1.
                                                 # This may not be what we want though as 1 would black out the entire image
                             }