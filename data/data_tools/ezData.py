'''
root/data/data_tools/ezData.py

Overview:

Rules:
'''

### packages
import os
import numpy as np
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



class ezData_Images(ezData):
    def __init__(self, data_dir=None, x=None, y=None, batch_size=None):
        '''
        Options:
         (1) load data straight into Augmentor.Pipeline if there a parent directory and all images of each class
            are in their own respective subdirectories
         (2) load in the data as np.arrays of x and y, and eventually manually feed into Augmentor.Pipeline
        '''
        globals()['Augmentor'] = importlib.import_module('Augmentor')
        globals()['PIL'] = importlib.import_module('PIL')
        
        if (data_dir is not None) and (os.path.isdir(data_dir)):
            self.option = 1
        elif (x is not None) and (y is not None):
            self.option = 2
        else:
            print("error")

        self.clear_pipeline()

        if batch_size is None:
            batch_size = len(self.x)
        self.batch_size = batch_size


    def clear_pipeline(self):
        '''
        Method clears the data structures so that individual can be re-evaluated
        '''
        if self.option==1:
            self.pipeline = Augmentor.Pipeline(source_directory=data_dir)
            #self.x = self.pipeline.augmentor_images
            #self.y = self.pipeline.class_labels

        elif self.option==2:
            self.pipeline = Augmentor.Pipeline()
            self.x = x
            self.y = y

        else:
            print("error")


    def get_next_batch(self):
        '''
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