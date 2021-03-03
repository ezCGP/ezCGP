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
from data.data_tools import ezData


### init dict
operator_dict = {}


def rotate(ez_augmentor, probability=.5, max_rotation=10):
    '''
    https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.rotate
    "This function will rotate **in place**, and crop the largest possible rectangle from the rotated image."

    prob: float (0,1]
    max_left/right_rotation: int [0,25]

    going to assign max_rotation to both left and right
    '''
    ez_augmentor.pipeline.rotate(probability=probability,
                                 max_left_rotation=max_rotation,
                                 max_right_rotation=max_rotation)
    return ez_augmentor

operator_dict[rotate] = {"inputs": [ezData.ezData_Augmentor],
                         "output": ezData.ezData_Augmentor,
                         "args": [argument_types.ArgumentType_LimitedFloat0to1,
                                  argument_types.ArgumentType_Int0to25]
                        }


def flip_left_right(ez_augmentor, probability=.5):
    '''
    https://arxiv.org/pdf/1912.11370v2.pdf
    https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.flip_left_right
    "Flip (mirror) the image along its horizontal axis, i.e. from left to right."
    prob: float (0,1]
    '''
    ez_augmentor.pipeline.flip_left_right(probability=probability)
    return ez_augmentor

operator_dict[flip_left_right] = {"inputs": [ezData.ezData_Augmentor],
                                  "output": ezData.ezData_Augmentor,
                                  "args": [argument_types.ArgumentType_LimitedFloat0to1]
                                 }


def flip_random(ez_augmentor, probability=.5):
    '''
    https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.flip_random
    "This function mirrors the image along either the horizontal axis or the vertical access.
    The axis is selected randomly."
    prob: float (0,1]
    '''
    ez_augmentor.pipeline.flip_random(probability=probability)
    return ez_augmentor

operator_dict[flip_random] = {"inputs": [ezData.ezData_Augmentor],
                              "output": ezData.ezData_Augmentor,
                              "args": [argument_types.ArgumentType_LimitedFloat0to1]
                             }


def flip_top_bottom(ez_augmentor, probability=.5):
    '''
    https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.flip_top_bottom
    "Flip (mirror) the image along its vertical axis, i.e. from top to bottom."
    prob: float (0,1]
    '''
    ez_augmentor.pipeline.flip_top_bottom(probability=probability)
    return ez_augmentor

operator_dict[flip_top_bottom] = {"inputs": [ezData.ezData_Augmentor],
                                  "output": ezData.ezData_Augmentor,
                                  "args": [argument_types.ArgumentType_LimitedFloat0to1]
                                 }


''' can't crop an image if using tf.keras.preprocessing.image.ImageDataGenerator.preprocessing_function since it expects image
# stay the same size
def crop_random(ez_augmentor, probability, percentage_area, randomise_percentage_area=False):
    ''
    https://arxiv.org/pdf/1912.11370v2.pdf
    https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.crop_random
    prob: float (0,1]
    percentage_area: float [0.1,1)
    randomise_percentage_area: bool -> if True will use given percentage_area as an upper bound and 0 as lower
    ''
    if percentage_area < 0.1:
        # here we are using argument_types.ArgumentType_LimitedFloat0to1 which goes from [0.05,1] in 0.05 increments so
        # we are being a bit lazy by using it and adjusting the lowerlimit
        percentage_area = 0.1

    ez_augmentor.pipeline.crop_random(probability=probability,
                                      percentage_area=percentage_area,
                                      randomise_percentage_area=randomise_percentage_area)
    return ez_augmentor

operator_dict[crop_random] = {"inputs": [ezData.ezData_Augmentor],
                              "output": ezData.ezData_Augmentor,
                              "args": [argument_types.ArgumentType_LimitedFloat0to1,
                                       argument_types.ArgumentType_LimitedFloat0to1,
                                       argument_types.ArgumentType_Bool]
                             }
'''


def invert(ez_augmentor, probability):
    '''
    "warning:: This function will cause errors if used on binary, 1-bit
     palette images (e.g. black and white)"
    prob: float (0,1]
    '''
    ez_augmentor.pipeline.invert(probability=probability)
    return ez_augmentor

operator_dict[invert] = {"inputs": [ezData.ezData_Augmentor],
                         "output": ezData.ezData_Augmentor,
                         "args": [argument_types.ArgumentType_LimitedFloat0to1]
                        }


def random_brightness(ez_augmentor, probability, factor0, factor1):
    '''
    prob: float (0,1]
    factor: float [0,inf) "The value  0.0 gives a black image, value 1.0 gives the
            original image, value bigger than 1.0 gives more bright image."
    '''
    #NOTE factors can be equal
    min_factor = min([factor0, factor1])
    max_factor = max([factor0, factor1])
    ez_augmentor.pipeline.random_brightness(probability=probability,
                                            min_factor=min_factor,
                                            max_factor=max_factor)
    return ez_augmentor

operator_dict[random_brightness] = {"inputs": [ezData.ezData_Augmentor],
                                    "output": ezData.ezData_Augmentor,
                                    "args": [argument_types.ArgumentType_LimitedFloat0to1,
                                             argument_types.ArgumentType_Float0to10,
                                             argument_types.ArgumentType_Float0to10]
                                   }


def random_color(ez_augmentor, probability, factor0, factor1):
    '''
    prob: float (0,1]
    factor: float [0,inf) "The value 0.0 gives a black and white image, value 1.0 gives the original image."
    '''
    #NOTE factors can be equal
    min_factor = min([factor0, factor1])
    max_factor = max([factor0, factor1])
    ez_augmentor.pipeline.random_color(probability=probability,
                                       min_factor=min_factor,
                                       max_factor=max_factor)
    return ez_augmentor

operator_dict[random_color] = {"inputs": [ezData.ezData_Augmentor],
                               "output": ezData.ezData_Augmentor,
                               "args": [argument_types.ArgumentType_LimitedFloat0to1,
                                        argument_types.ArgumentType_Float0to10,
                                        argument_types.ArgumentType_Float0to10]
                              }


def random_contrast(ez_augmentor, probability, factor0, factor1):
    '''
    prob: float (0,1]
    factor: float [0,inf) "The value  0.0 gives s solid grey image, value 1.0 gives the original image."
    '''
    #NOTE factors can be equal
    min_factor = min([factor0, factor1])
    max_factor = max([factor0, factor1])
    ez_augmentor.pipeline.random_contrast(probability=probability,
                                          min_factor=min_factor,
                                          max_factor=max_factor)
    return ez_augmentor 

operator_dict[random_contrast] = {"inputs": [ezData.ezData_Augmentor],
                                  "output": ezData.ezData_Augmentor,
                                  "args": [argument_types.ArgumentType_LimitedFloat0to1,
                                           argument_types.ArgumentType_Float0to10,
                                           argument_types.ArgumentType_Float0to10]
                                 }


def random_distortion(ez_augmentor, probability, grid_width, grid_height, magnitude):
    '''
    "Performs a random, elastic distortion on an image...
    
    *Good* values for parameters are between 2 and 10 for the grid
        width and height, with a magnitude of between 1 and 10. Using values
        outside of these approximate ranges may result in unpredictable
        behaviour."
    
    prob: float (0,1]
    grid_width/height: int [2,10] -> going to use int1to10 so gotta force to 2to10
    magnitude: int [1,10]
    '''
    if grid_width < 2:
        grid_Width = 2
    if grid_height < 2:
        grid_height = 2
    ez_augmentor.pipeline.random_distortion(probability=probability,
                                            grid_width=grid_width,
                                            grid_height=grid_height,
                                            magnitude=magnitude)
    return ez_augmentor

operator_dict[random_distortion] = {"inputs": [ezData.ezData_Augmentor],
                                    "output": ezData.ezData_Augmentor,
                                    "args": [argument_types.ArgumentType_LimitedFloat0to1,
                                             argument_types.ArgumentType_Int1to10,
                                             argument_types.ArgumentType_Int1to10,
                                             argument_types.ArgumentType_Int1to10]
                                   }


def random_erasing(ez_augmentor, probability, rectangle_area):
    '''
    prob: float (0,1]
    rectangle_area: The percentage area of the image to occlude
         with the random rectangle, between (0.1, 1.]
    '''
    if rectangle_area <= 0.1:
        # assuming we are using ArgumentType_LimitedFloat0to1 so lowest value is 0
        rectangle_area += 0.1
    ez_augmentor.pipeline.random_erasing(probability=probability,
                                         rectangle_area=rectangle_area)
    return ez_augmentor

operator_dict[random_erasing] = {"inputs": [ezData.ezData_Augmentor],
                                 "output": ezData.ezData_Augmentor,
                                 "args": [argument_types.ArgumentType_LimitedFloat0to1,
                                          argument_types.ArgumentType_LimitedFloat0to1]
                                }


def rotate_random_90(ez_augmentor, probability):
    '''
    Rotate an image by either 90, 180, or 270 degrees, selected randomly.
    
    prob: float (0,1]
    '''
    ez_augmentor.pipeline.rotate_random_90(probability=probability)
    return ez_augmentor

operator_dict[rotate_random_90] = {"inputs": [ezData.ezData_Augmentor],
                                   "output": ezData.ezData_Augmentor,
                                   "args": [argument_types.ArgumentType_LimitedFloat0to1]
                                  }


def shear(ez_augmentor, probability, max_shear):
    '''
    "Shear the image by a specified number of degrees.
    In practice, shear angles of more than 25 degrees can cause
        unpredictable behaviour."
    
    prob: float (0,1]
    max_shear: [0, 25]
    '''
    ez_augmentor.pipeline.shear(probability=probability,
                                max_shear_left=max_shear,
                                max_shear_right=max_shear)
    return ez_augmentor

operator_dict[shear] = {"inputs": [ezData.ezData_Augmentor],
                        "output": ezData.ezData_Augmentor,
                        "args": [argument_types.ArgumentType_LimitedFloat0to1,
                                 argument_types.ArgumentType_Int0to25]
                       }


def skew(ez_augmentor, probability, magnitude):
    '''
    "Skew an image in a random direction, either left to right,
        top to bottom, or one of 8 corner directions."
    
    prob: float (0,1]
    magnitude: float (0,1] ...documentation says min 0.1 in comments but 0 in code
    '''
    ez_augmentor.pipeline.skew(probability=probability,
                               magnitude=magnitude)
    return ez_augmentor

operator_dict[skew] = {"inputs": [ezData.ezData_Augmentor],
                       "output": ezData.ezData_Augmentor,
                       "args": [argument_types.ArgumentType_LimitedFloat0to1,
                                argument_types.ArgumentType_LimitedFloat0to1]
                      }

