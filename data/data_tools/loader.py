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
        return ezdata.ezData_numpy(np.array(x), np.array(y))
    
    
    
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
        train_data = ezdata.ezData_Images(*train_xy)
        validate_data = ezdata.ezData_Images(*validate_xy)
        test_data = ezdata.ezData_Images(*test_xy)'''
        
        # load in images
        train_images = ezdata.ezData_Images(*train_xy)
        validate_images = ezdata.ezData_Images(*validate_xy)
        test_images = ezdata.ezData_Images(*test_xy)
        
        # load in Augmentor pipeline
        train_augmentor = ezdata.ezData_Augmentor()
        validate_augmentor = ezdata.ezData_Augmentor()
        test_augmentor = ezdata.ezData_Augmentor()

        # call steal_metadata() to grab a few attributes from ezData_Images() class over
        train_augmentor.steal_metadata(train_images)
        validate_augmentor.steal_metadata(validate_images)
        test_augmentor.steal_metadata(test_images)

        return ([train_images, train_augmentor],
                [validate_images, validate_augmentor],
                [test_images, test_augmentor])



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
        train_data = ezdata.ezData_Images(x_train, y_train)
        validate_data = ezdata.ezData_Images(x_test, y_test)
        test_datapair = []
        return train_data, validate_data, test_datapair



class ezDataLoader_MNIST(ezDataLoader):
    def __init__(self,
                 train_split=5/6,
                 validate_split=1/6,
                 test_split=0):
        '''
        These weights are not the actual split of the dataset but how we want to
        split the original train-set provided by tf.keras so we can get train,
        validate, and test sets.
        '''
        super().__init__(train_split, validate_split, test_split)


    def load(self):
        '''
        https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
        Image shape is (Samples, width, height, channels)

        copy/paste:
        x_train: uint8 NumPy array of grayscale image data with shapes (60000, 28, 28),
                containing the training data. Pixel values range from 0 to 255.
        y_train: uint8 NumPy array of digit labels (integers in range 0-9) with shape
                (60000,) for the training data.
        x_test: uint8 NumPy array of grayscale image data with shapes (10000, 28, 28),
                containing the test data. Pixel values range from 0 to 255.
        y_test: uint8 NumPy array of digit labels (integers in range 0-9) with shape
                (10000,) for the test data.

        Going to further split the train set they gave us into train and validate.

        See Issue #288 on why we can't use Augmentor for MNIST or any other single-
        channel image dataset with tf.keras.
        '''
        import tensorflow.keras.datasets.mnist as mnist_dataset
        (x_train_ori, y_train_ori), (x_test, y_test) = mnist_dataset.load_data()
        (x_train, y_train), (x_validate, y_validate), _ = self.split(x_train_ori, y_train_ori)

        # mnist only has 1 channel images and that dim is not included in the shape but
        # we need that dim for our convolutions...gonna add here
        x_train = np.expand_dims(x_train, axis=-1)
        x_validate = np.expand_dims(x_validate, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        '''
        # convert to [0,1] floats from [0,255] uint8
        x_train = (x_train/(2**8-1)).astype(np.float32)
        x_validate = (x_validate/(2**8-1)).astype(np.float32)
        x_test = (x_test/(2**8-1)).astype(np.float32)
        '''
        # convert to floats
        x_train = x_train.astype(np.float32)
        x_validate = x_validate.astype(np.float32)
        x_test = x_test.astype(np.float32)

        # convert single array of labels into 01matrix
        def make_matrix_label(array_label):
            # assuming labels go from [0,N-1]
            matrix_label = np.zeros(shape=(len(array_label), array_label.max()+1), dtype=array_label.dtype)
            for row, col in enumerate(array_label):
                matrix_label[row,col]=1
            return matrix_label

        y_train = make_matrix_label(y_train)
        y_validate = make_matrix_label(y_validate)
        y_test = make_matrix_label(y_test)

        train_images = ezdata.ezData_Images(x_train, y_train)
        validate_images = ezdata.ezData_Images(x_validate, y_validate)
        test_images = ezdata.ezData_Images(x_test, y_test)

        return ([train_images],
                [validate_images],
                [test_images])



class ezDataLoader_EMADE_Titanic(ezDataLoader):
    '''
    going to try and mimic how EMADE loads in DataPairs
    for Titanic dataset

    assuming we used the emade.sh script and downloaded emade repo into datasets dir
    '''
    def __init__(self):
        super().__init__(1,0,0)


    def load(self):
        '''
        using the filepaths as it's read from the config xml in emade
        https://github.gatech.edu/emade/emade/blob/CacheV2/templates/input_titanic.xml
        '''
        train_datapair = ezdata.ezData_EMADE(train_filenames=['datasets/titanic/train_0.csv.gz',
                                                              'datasets/titanic/train_1.csv.gz',
                                                              'datasets/titanic/train_2.csv.gz',
                                                              'datasets/titanic/train_3.csv.gz',
                                                              'datasets/titanic/train_4.csv.gz'],
                                             test_filenames=['datasets/titanic/test_0.csv.gz',
                                                             'datasets/titanic/test_1.csv.gz',
                                                             'datasets/titanic/test_2.csv.gz',
                                                             'datasets/titanic/test_3.csv.gz',
                                                             'datasets/titanic/test_4.csv.gz'],
                                             dtype='featuredata')
        validate_datapair = []
        test_datapair = []

        return train_datapair, validate_datapair, test_datapair



class ezDataLoader_SyscoSearch(ezDataLoader):
    '''
    We want to build out a dictionary for each individual/experiment and then fill a list of them and write to json.
    This will just initialize the dictionary for each individual.
    '''
    def __init__(self):
        super().__init__(1,0,0)


    def load(self):
        experiment_hyperparams = ezdata.ezData_dict()
        experiment_hyperparams['locale'] = 'en_US'
        experiment_hyperparams['size'] = 50
        experiment_hyperparams['from'] = 0
        experiment_hyperparams['titleLocaleBoost'] = -1
        experiment_hyperparams['descriptionLocaleBoost'] = -1
        experiment_hyperparams['brandNameLocaleBoost'] = -1
        experiment_hyperparams['categoryIntermediateNameLocaleBoost'] = -1
        experiment_hyperparams['lineDescriptionLocaleBoost'] = -1
        experiment_hyperparams['stockTypeSBoost'] = -1
        experiment_hyperparams['stockTypeRBoost'] = -1
        experiment_hyperparams['stockTypeDBoost'] = -1
        experiment_hyperparams['isSyscoBrandBoost'] = -1
        experiment_hyperparams['imageExistsBoost'] = -1
        experiment_hyperparams['CustomerCountFactorBoost'] = 2.5
        return ([experiment_hyperparams], None, None)


ting = ezDataLoader_SyscoSearch()