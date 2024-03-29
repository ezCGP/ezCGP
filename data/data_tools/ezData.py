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
    def __init__(self, x, y=None):
        self.x = x
        self.y = y
        # in individual_evaluate, we'll deepcopy the datalist to ensure that the global/original data doesn't get altered
        # but sometimes we don't want to deepcopy, like for images because it would take up too much ram and we don't alter images
        self.do_not_deepcopy = False



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
        self.do_not_deepcopy = True # takes up too much RAM unless other wise decided



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

