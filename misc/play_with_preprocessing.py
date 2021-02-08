'''
play with importing primitives from augmentor preprocessing and make sure they don't error

one thing to look out for is if the images change shape

as of writing this, we use keras_preprocessing_func from Augmentor to wrap all the operators together after we already add them to the pipeline. So we may want to mimic that for ease of debugging.
https://github.com/ezCGP/ezCGP/blob/2020F-BaseCodeDevelopment/codes/block_definitions/block_evaluate.py#L429
https://augmentor.readthedocs.io/en/master/_modules/Augmentor/Pipeline.html#Pipeline.keras_preprocess_func
'''

### packages
import os
import numpy as np
from copy import deepcopy
import inspect
import pdb
#import functools
import Augmentor

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from codes.block_definitions.utilities import operators_Augmentor_preprocessing as ops
#from codes import block_operations


### create some fake data
batch_size = 20
image_shape = (256, 256, 3)
images = np.random.random((batch_size,)+image_shape)
#images = np.random.randint(0, 256, (batch_size,)+(image_shape)).astype(np.uint8)
labels = np.random.randint(0, 2, (batch_size,))


### create pipeline
pipeline = Augmentor.Pipeline()
primtives = []
for name, execute in inspect.getmembers(globals()['ops']): # returns list of tuples of everything in that module
    if (inspect.isfunction(execute)) and  (execute.__module__.endswith('operators_Augmentor_preprocessing')) and (execute in ops.operator_dict):
        print(name, execute)
        # check if what we are pulling is a function, then make sure it is a function defined in that module
        # as oposed to something imported like dirname from os.path
        primtives.append(execute)
        
        # build out args
        args = []
        for arg_type in ops.operator_dict[execute]["args"]:
            arg_instance = arg_type()
            args.append(arg_instance.value)
        
        pipeline = execute(pipeline, *args)


### mimic Augmentor.Pipeline.keras_preprocess_func
def mimic_keras_preprocess_func(pipeline, image):
    # https://augmentor.readthedocs.io/en/master/_modules/Augmentor/Pipeline.html#Pipeline.keras_preprocess_func
    image = Augmentor.Operations.Image.fromarray(np.uint8(255 * image))
    for operation in pipeline.operations:
        print("Executing %s" % type(operation).__name__) # operation is a class not the actual method
        
        r = Augmentor.Operations.random.uniform(0, 1)
        if r < operation.probability:
            try:
                image = operation.perform_operation([image])[0]
            except Exception as err:
                print("ehhh yikes: %s" % err)
                pdb.set_trace()
    return image
    
    


mimic_keras_preprocess_func(pipeline, images[0])
