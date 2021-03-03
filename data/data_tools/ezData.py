'''
root/data/data_tools/ezData.py

Overview:

Rules:
'''

### packages
import os
import numpy as np
import importlib
'''
### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__))))) 

### absolute imports wrt root
'''


class ezData():
    def __init__(self, x, y=None):
        self.x = x
        self.y = y



class ezData_float(np.float64, ezData):
    '''
    breaks if you inherit ezData first
    NOTE: this actually doesn't seem to work in making isinstance(instance, ezData_float) True

    Good enough for symbolic regression though.
    '''
    def __init__(self, x):
        pass


    def __new__(cls, x):
        instance = np.float64(x).view(cls)
        return instance



class ezData_numpy(ezData, np.ndarray):
    '''
    see root/misc/inherit_npndarray.py for what started all this

    for now will keep x attribute as redundant way to access data, but never explored to see
    how much of an effect this had on the size of the class by self referencing

    WARNING: when you deepcopy an ezData_numpy instance, it looses all it's attributes like x and y
    '''
    def __init__(self, x, y):
        '''
        anything in here will happen after __new__ call
        '''
        pass


    def __new__(cls, x, y=None):
        instance = np.asarray(x).view(cls)
        instance.x = instance
        instance.y = y
        return instance



class ezData_Images(ezData):
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        self.num_images = len(self.x)
        self.num_classes = np.unique(self.y, axis=0).shape[0]
        self.image_shape = self.x[0].shape



class ezData_Augmentor(ezData):
    def __init__(self, data_dir=None):
        '''
        always assume: BatchSize x Height x Width x Channels
        Options:
         (1) load data straight into Augmentor.Pipeline if there a parent directory and all images of each class
            are in their own respective subdirectories
         (2) load in the data as np.arrays of x and y, and eventually manually feed into Augmentor.Pipeline
        '''
        globals()['Augmentor'] = importlib.import_module('Augmentor')
        #globals()['PIL'] = importlib.import_module('PIL')
        
        if (data_dir is not None) and (os.path.isdir(data_dir)):
            self.option = 1
            self.pipeline = Augmentor.Pipeline(source_directory=data_dir)
            self.num_images = len(self.pipeline.augmentor_images)
            self.num_classes = np.unique(self.pipeline.class_labels, axis=0).shape[0]
            self.image_shape = (0,) # TODO get
        else:
            self.option = 2
            self.pipeline = Augmentor.Pipeline()


    def steal_metadata(self, ez_images_object: ezData_Images):
        '''
        bit of a cheat...throw some useful image_wrapper attributes to pipeline_wrapper.
        say, when we are doing transfer learning and building the graph, we don't need all the images
        but only need the image shape, so it would make sense to pass in only the pipeline object with
        the image shape to the transfer learning block...more lightweight
        '''
        self.image_shape = ez_images_object.image_shape
        self.num_classes = ez_images_object.num_classes
        self.num_images = ez_images_object.num_images



class ezData_Time(ezData):
    pass
