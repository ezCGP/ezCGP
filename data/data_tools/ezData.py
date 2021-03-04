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
    def __init__(self, x, y):
        self.x = x
        self.y = y



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



class ezData_Images(ezData):
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        self.num_images = len(self.x)
        self.num_classes = np.unique(self.y, axis=0).shape[0]
        self.image_shape = self.x[0].shape



class ezData_AugmentorImages(ezData):
    '''
    the motivation here is to separate out pipeline from images so that
    after we evaluate an individual, we can save the output of each block, and 
    compress the individual to a pkl file without it blowing up in size. if we
    have to pass the data object through the preprocessing + augmentation blocks
    so that we can decorate the pipeline, we don't want to have to drag around 
    the images too and then have the images saved to the pkl with the final
    pipeline. this way we separate them, and then also make a custom
    IndividualEvaluate class that passes only the pipeline through the frist
    few blocks

    here we assume that the ezData_Images() and ez_Augmentor() objects have
    already been created and are just going to get passed in at init
    '''
    def __init__(self, ez_images_object, ez_augmentor_object):
        self.images_wrapper = ez_images_object
        self.pipeline_wrapper = ez_augmentor_object

        # bit of a cheat...throw some useful image_wrapper attributes to pipeline_wrapper.
        # say, when we are doing transfer learning and building the graph, we don't need all the images
        # but only need the image shape, so it would make sense to pass in only the pipeline object with
        # the image shape to the transfer learning block...more lightweight
        self.pipeline_wrapper.image_shape = ez_images_object.image_shape
        self.pipeline_wrapper.num_classes = ez_images_object.num_classes
        self.pipeline_wrapper.num_images = ez_images_object.num_images



class ezData_Images_depreciated(ezData):
    def __init__(self, x=None, y=None, data_dir=None):
        '''
        always assume: BatchSize x Height x Width x Channels
        Options:
         (1) load data straight into Augmentor.Pipeline if there a parent directory and all images of each class
            are in their own respective subdirectories
         (2) load in the data as np.arrays of x and y, and eventually manually feed into Augmentor.Pipeline
        '''
        globals()['Augmentor'] = importlib.import_module('Augmentor')
        globals()['PIL'] = importlib.import_module('PIL')
        
        if (data_dir is not None) and (os.path.isdir(data_dir)):
            self.option = 1
            self.pipeline = Augmentor.Pipeline(source_directory=data_dir)
            self.x = None #self.pipeline.augmentor_images
            self.y = None #self.pipeline.class_labels
            self.num_images = len(self.pipeline.augmentor_images)
            self.num_classes = np.unique(self.pipeline.class_labels, axis=0).shape[0] # TODO check
            self.image_shape = (0,) # TODO get
        elif (x is not None) and (y is not None):
            self.option = 2
            self.pipeline = Augmentor.Pipeline()
            self.x = x
            self.y = y
            self.num_images = len(self.x)
            self.num_classes = 10 #np.unique(self.y, axis=0).shape[0] # TODO switch back to np.unique after done debug
            self.image_shape = self.x[0].shape
        else:
            print("error")


    def get_next_batch(self):
        '''
        Likely going to abandon these methods and use tf.keras.preprocessing.image.ImageDataGenerator
        ----------
        When using Option (2)

        Applies augmentation and preprocessing pipeline to a batch of data
            - Use this in evaluator methods to train a network
        batch_size: amount of data to sample for x_train, and y_train
        returns: x_batch, y_batch -> len(x_batch) = batch_size
        '''
        # augmentation pipeline
        index = np.random.choice(np.arange(len(self.x)),  self.batch_size)
        x_batch = [PIL.Image.fromarray(img) for img in self.x[index]]
        y_batch = self.y[index]

        augmentor_method = self.pipeline.torch_transform()
        x_batch = np.array([np.asarray(augmentor_method(x)) for x in x_batch])

        return x_batch, y_batch


    def get_generator(self):
        '''
        when using Option (1)
        '''
        pass


class ezData_Time(ezData):
    pass
