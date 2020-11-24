'''
root/data/data_tools/loader.py

Overview:

Rules:
'''

### packages
import os
import numpy as np
import importlib
from abc import ABC, abstractmethod
import six
from six.moves import cPickle as pickle

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__))))) 

### absolute imports wrt root
import data.data_tools.ezData as ezdata


class ezDataLoader(ABC):
    def __init__(self,
                 train_split=0.5,
                 validate_split=0.25,
                 test_split=0.25):
        if train_split + validate_split + test_split > 1:
            train_split=0.5
            validate_split=0.25
            test_split=0.25

        self.train_split = train_split
        self.validate_split = validate_split
        self.test_split = test_split


    @abstractmethod
    def load(self):
        pass


    def split(self, x, y=None):
        train_count = int(len(x) * self.train_split)
        validate_count = int(len(x) * self.validate_split)
        test_count = len(x) - train_count - validate_count

        index = np.arange(len(x))
        train_index = np.random.choice(index, size=train_count, replace=False)
        remaining_index = np.delete(index, np.sort(train_index))
        validate_index = np.random.choice(remaining_index, size=validate_count, replace=False)
        test_index = np.delete(index, np.sort(np.hstack([train_index,validate_index])))

        x_train = x[train_index]
        x_validate = x[validate_index]
        x_test = x[test_index]

        if y is None:
            y_train = None
            y_validate = None
            y_test = None
        else:
            y_train = y[train_index]
            y_validate = y[validate_index]
            y_test = y[test_index]

        return (x_train, y_train), (x_validate, y_validate), (x_test, y_test)



class ezDataLoader_numpy(ezDataLoader):
    '''
    just any simple (x,y) data as numpy arrays.
    assuming only one dataset...no splitting
    '''
    def __init__(self):
        super().__init__(1,0,0)


    def load(self, x, y):
        return ezdata.ezData(np.array(x), np.array(y))
    
    
    
class ezDataLoader_CIFAR10(ezDataLoader):
    '''
    the way we download the cifar data, it doesn't download
    single png images. it downloads massive 5 30+MB files, so
    we can't initialize Augmentor.Pipeline with a folder of images
    '''
    def __init__(self,
                 train_split=0.5,
                 validate_split=0.25,
                 test_split=0.25):
        super().__init__(train_split, validate_split, test_split)
        self.data_dir = os.path.join(os.path.dirname(__file__), '../datasets/cifar10/cifar-10-batches-py')
        globals()['tf'] = importlib.import_module('tensorflow')


    def load(self):
        data = []
        image_shape = (10000, 3, 32, 32)  # (Samples, channels, width, height)
        for ith_file in range(1, 6):
            filepath = os.path.join(self.data_dir, 'data_batch_%i' % (ith_file))
            with open(filepath, 'rb') as f:
                if six.PY2:
                    datadict = pickle.load(f)
                elif six.PY3:
                    datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(*image_shape).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
            data.append((X,Y))

        x = np.concatenate([x[0] for x in data]).astype(np.uint8) # 2020S code eventually calls PIL.Image.fromarray; I don't think needed
        y = np.concatenate([x[1] for x in data])
        y = tf.keras.utils.to_categorical(y, num_classes=10)

        train_xy, validate_xy, test_xy= self.split(x, y)

        # control the input dataset
        ''' old
        train_datapair = ezdata.ezData_Images(*train_xy)
        validate_datapair = ezdata.ezData_Images(*validate_xy)
        test_datapair = ezdata.ezData_Images(*test_xy)'''
        
        # load in images
        train_images = ezdata.ezData_Images(*train_xy)
        validate_images = ezdata.ezData_Images(*validate_xy)
        test_images = ezdata.ezData_Images(*test_xy)
        
        # load in Augmentor pipeline
        train_augmentor = ezdata.ezData_Augmentor()
        validate_augmentor = ezdata.ezData_Augmentor()
        test_augmentor = ezdata.ezData_Augmentor()
        
        # load together
        train_datapair = ezdata.ezData_AugmentorImages(train_images, train_augmentor)
        validate_datapair = ezdata.ezData_AugmentorImages(validate_images, validate_augmentor)
        test_datapair = ezdata.ezData_AugmentorImages(test_images, test_augmentor)

        return train_datapair, validate_datapair, test_datapair



class ezDataLoader_CIFAR10_old(ezDataLoader):
    '''
    the way we download the cifar data, it doesn't download
    single png images. it downloads massive 5 30+MB files, so
    we can't initialize Augmentor.Pipeline with a folder of images
    '''
    def __init__(self,
                 train_split=0.5,
                 validate_split=0.25,
                 test_split=0.25):
        super().__init__(train_split, validate_split, test_split)
        self.data_dir = os.path.join(os.path.dirname(__file__), '../datasets/cifar10/cifar-10-batches-py')
        globals()['tf'] = importlib.import_module('tensorflow')


    def load(self):
        data = []
        image_shape = (10000, 3, 32, 32)  # (Samples, channels, width, height)
        for ith_file in range(1, 6):
            filepath = os.path.join(self.data_dir, 'data_batch_%i' % (ith_file))
            with open(filepath, 'rb') as f:
                if six.PY2:
                    datadict = pickle.load(f)
                elif six.PY3:
                    datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(*image_shape).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
            data.append((X,Y))

        x = np.concatenate([x[0] for x in data]).astype(np.uint8) # 2020S code eventually calls PIL.Image.fromarray; I don't think needed
        y = np.concatenate([x[1] for x in data])
        y = tf.keras.utils.to_categorical(y, num_classes=10)

        train_xy, validate_xy, test_xy= self.split(x, y)

        # control the input dataset
        train_datapair = ezdata.ezData_Images(*train_xy)
        validate_datapair = ezdata.ezData_Images(*validate_xy)
        test_datapair = ezdata.ezData_Images(*test_xy)

        return train_datapair, validate_datapair, test_datapair



class ezDataLoader_CIFAR100(ezDataLoader):
    def __init__(self,
                 train_split=0.5,
                 validate_split=0.25,
                 test_split=0.25):
        super().__init__(train_split, validate_split, test_split)

    def load(self):
        # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar100/load_data
        ###
        # 50,000 32x32 color training images and 10,000 test images
        # labeled over 100 fine-grained classes that are grouped into 20 coarse-grained classes.
        # train_split ratio is not
        # want channel last 
        ###
        import tensorflow as tf
        cifar100 = tf.keras.datasets.cifar100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        train_datapair = ezdata.ezData_Images(x_train, y_train)
        validate_datapair = ezdata.ezData_Images(x_test, y_test)
        return train_datapair, validate_datapair



class ezDataLoader_MNIST(ezDataLoader):
    def __init__(self,
                 train_split=0.5,
                 validate_split=0.25,
                 test_split=0.25):
        super().__init__(train_split, validate_split, test_split)

    def load(self):
        # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
        # image shape is (Samples, width, height, channels)
        import tensorflow as tf
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        train_datapair = ezdata.ezData_Images(x_train, y_train)
        validate_datapair = ezdata.ezData_Images(x_test, y_test)
        return train_datapair, validate_datapair