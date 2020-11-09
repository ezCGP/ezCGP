'''
root/data/data_tools/ezData.py

Overview:

Rules:
'''

### packages
import os
import sys
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



class ezData_Images(ezData):
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
            self.num_classes = np.unique(self.class_labels, axis=0).shape[0] # TODO check
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



class ezData_EMADE(ezData):
    def __init__(self,
                 train_filenames,
                 test_filenames,
                 dtype,
                 use_cache=False,
                 compress=False):
        '''
        https://github.gatech.edu/emade/emade/blob/CacheV2/src/GPFramework/EMADE.py#L318
        '''
        data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        emade_dir = os.path.join(data_dir, 'datasets', 'emade')
        emade_src_dir = os.path.join(emade_dir, 'src')
        sys.path.append(emade_src_dir)
        from GPFramework.data import EmadeDataPair

        if dtype == 'featuredata':
            from GPFramework.data import load_feature_data_from_file
            load_function = load_feature_data_from_file
        elif dtype == 'streamdata':
            from GPFramework.data import load_many_to_one_from_file
            load_function = load_many_to_one_from_file
        else:
            print("wrong dtype given", dtype)
            exit()

        def reduce_instances(emadeDataTuple):
            emadeData, cache = emadeDataTuple
            return emadeData, cache

        print(emade_dir, train_filenames[0])
        train_data_array = [reduce_instances(load_function(os.path.join(emade_dir, folder),
                                                           use_cache=use_cache,
                                                           compress=compress))
                                for folder in train_filenames]

        test_data_array = [reduce_instances(load_function(os.path.join(emade_dir, folder),
                                                          use_cache=use_cache,
                                                          compress=compress,
                                                          hash_data=True))
                               for folder in test_filenames]


        # Copy the truth data in to its own location
        truth_data_array = [test_data[0].get_target() for test_data in test_data_array]

        # Clear out the truth data from the test data
        [test_data[0].set_target(np.full(test_data[0].get_target().shape,np.nan)) for test_data in test_data_array]

        # Stores DataPair object
        self.x = [EmadeDataPair(train_data, test_data)
                    for train_data, test_data in zip(train_data_array, test_data_array)]
        self.y = None

