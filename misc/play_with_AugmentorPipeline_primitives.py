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
import importlib
import inspect
import pdb
import Augmentor

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from codes.block_definitions.utilities import operators_Augmentor_augmentation as ops_0
from codes.block_definitions.utilities import operators_Augmentor_preprocessing as ops_1


### create some fake data
def create_fake_data(batch_size=20,
                     image_shape=(256, 256, 3),
                     asint=True):
    np.random.seed(20)
    if asint:
        images = np.random.randint(0, 256, (batch_size,)+(image_shape)).astype(np.uint8)
    else:
        images = np.random.random((batch_size,)+image_shape)

    labels = np.random.randint(0, 2, (batch_size,))
    
    return images, labels


### create pipeline
def create_pipeline():
    pipeline = Augmentor.Pipeline()
    operator_dict = {}
    primtives = []
    for module in ["operators_Augmentor_augmentation",
                   "operators_Augmentor_preprocessing"]:
        globals()[module] = importlib.import_module("codes.block_definitions.utilities.%s" % module)
        operator_dict.update(globals()[module].operator_dict)
        for name, execute in inspect.getmembers(globals()[module]): # returns list of tuples of everything in that module
            if (inspect.isfunction(execute)) and  (execute.__module__.endswith(module)) and (execute in operator_dict):
                # check if what we are pulling is a function, then make sure it is a function defined in that module
                # as oposed to something imported like dirname from os.path
                primtives.append(execute)

                # build out args
                args = []
                for arg_type in operator_dict[execute]["args"]:
                    arg_instance = arg_type()
                    args.append(arg_instance.value)

                pipeline = execute(pipeline, *args)

    return pipeline


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
    
    

if __name__ == "__main__":
    np.random.seed(20)
    
    pipeline = create_pipeline()
    images, labels = create_fake_data(asint=False)
    
    for image in images:
        image = mimic_keras_preprocess_func(pipeline, image)
        print("Final Shape of Image:", np.array(image).shape)
        break
