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

Rules:
Since we are manipulating an object instead of actually evaluating something, I decided to deepcopy the object
before returning it. In hindsight this shouldn't make a difference to the final result but it could make debugging
easier if the output from each node ends up being different things instead of different variables pointing to the same object.
'''

### packages
import Augmentor
from Augmentor.Operations import Operation
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



class Normalize(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability=1):
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability=probability)

    # Your class must implement the perform_operation method:
    def perform_operation(self, images):
        # Return the image so that it can further processed in the pipeline:
        augmentedImages = []
        for image in images:
            modImage = np.asarray(image) / 255.0
            augmentedImages.append(modImage)
        return augmentedImages


def normalize(pipeline):
    pipeline.add_operation(Normalize())
    return deepcopy(pipeline)

operator_dict[normalize] = {"inputs": [Augmentor.Pipeline],
                            "output": Augmentor.Pipeline,
                            "args": [] # TODO: no argument because we always want prob=1 right?
                           }