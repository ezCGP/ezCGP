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
from codes.block_definitions.utilities import argument_types
from codes.utilities.custom_logging import ezLogging



### init dict
operator_dict = {}


def ting(input, arg):
    pass

operator_dict[ting] = {"inputs": [datatype0],
                       "output": datatype0,
                       "args": argument_types.argtype0
                      }


