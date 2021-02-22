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
        train_datapair = ezdata.ezData_Images_depreciated(*train_xy)
        validate_datapair = ezdata.ezData_Images_depreciated(*validate_xy)
        test_datapair = ezdata.ezData_Images_depreciated(*test_xy)

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



class ezDataLoader_MNIST_TF(ezDataLoader):
    '''
    downloading mnist dataset with tensorflow
    '''
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



class ezDataLoader_MNIST(ezDataLoader):
    '''
    another mnist dataloader but without tensorflow.
    good if not using neural networks

    resource:
    http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/

     - Training set images: train-images-idx3-ubyte.gz (9.9 MB, 47 MB unzipped, and 60,000 samples)
     - Training set labels: train-labels-idx1-ubyte.gz (29 KB, 60 KB unzipped, and 60,000 labels)
     - Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 7.8 MB, unzipped and 10,000 samples)
     - Test set labels: t10k-labels-idx1-ubyte.gz (5 KB, 10 KB unzipped, and 10,000 labels)
    All images are (784,) arrays flattened from 28x28 images
    '''
    def __init__(self,
                 train_split=5/7,
                 validate_split=1/7,
                 test_split=1/7):
        super().__init__(train_split, validate_split, test_split)
        self.data_dir = os.path.join(os.path.dirname(__file__), '../datasets/mnist')


    def load(self):
        '''
        using mlxtend to deal with opeing the data
        http://rasbt.github.io/mlxtend/installation/
        use $ conda/pip install mlxtend

        see 'resource' in class documentation
        '''
        from mlxtend.data import loadlocal_mnist
        import platform
        if not platform.system() == 'Windows':
            x_train, y_train = loadlocal_mnist(
                                images_path=os.path.join(self.data_dir, 'train-images-idx3-ubyte'), 
                                labels_path=os.path.join(self.data_dir, 'train-labels-idx1-ubyte'))
            x_test, y_test = loadlocal_mnist(
                                images_path=os.path.join(self.data_dir, 't10k-images-idx3-ubyte'), 
                                labels_path=os.path.join(self.data_dir, 't10k-labels-idx1-ubyte'))

        else:
            x_train, y_train = loadlocal_mnist(
                                images_path=os.path.join(self.data_dir, 'train-images.idx3-ubyte'), 
                                labels_path=os.path.join(self.data_dir, 'train-labels.idx1-ubyte'))
            x_test, y_test = loadlocal_mnist(
                                images_path=os.path.join(self.data_dir, 't10k-images.idx3-ubyte'), 
                                labels_path=os.path.join(self.data_dir, 't10k-labels.idx1-ubyte'))

        # currently train 6/7 of data, test 1/7, but we need validation
        # so take 1/6 of train to make into validation
        validation_indx = np.random.choice(np.arange(x_train.shape[0]), size=x_train.shape[0]//6, replace=False)
        x_val = x_train[validation_indx]
        y_val = y_train[validation_indx]
        x_train = np.delete(x_train, validation_indx, axis=0)
        y_train = np.delete(y_train, validation_indx, axis=0)

        train_datapair = ezdata.ezData_Images(x_train, y_train)
        validate_datapair = ezdata.ezData_Images(x_val, y_val)
        test_datapair = ezdata.ezData_Images(x_test, y_test)

        return train_datapair, validate_datapair, test_datapair



class ezDataLoader_EmadeData(ezDataLoader):
    '''
    mimic emade.GPFramework.data.load_feature_data_from_file()
    but where we have already split our data into x and y for train, validate, test
    '''
    def __init__(self):
        super().__init__(1, 1, 1)
        self.data_dir = ""


    def load(self, ezDataLoaderClass, **kwargs):
        '''
        pass in the loader we would normally use to load the data, and 
        instead we'll use that to load the data into an EmadeData Object
        '''
        # first import emade stuff
        data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        emade_dir = os.path.join(data_dir, 'datasets', 'emade')
        emade_src_dir = os.path.join(emade_dir, 'src')
        sys.path.append(emade_src_dir)
        from GPFramework import data as emade_data

        ezcgp_data_list = ezDataLoaderClass(**kwargs).load() # list is -> [train datapair, validate datapair, test datapair]
        emade_data_list = []

        # mimic emade.GPFramework.data.load_feature_data_from_file()
        # https://github.gatech.edu/emade/emade/blob/CacheV2/src/GPFramework/data.py#L88
        def load_function(ezcgp_data):
            feature_array = []
            label_list = []
            points = []
            for feature, label in zip(ezcgp_data.x, ezcgp_data.y):
                class_data = np.array([np.float(label)])
                label_list.append(class_data)
                feature_data = np.array([feature], dtype='d')
                feature_array.append(feature_data)

                point = emade_data.EmadeDataInstance(target=class_data)
                point.set_stream(
                    emade_data.StreamData(np.array([[]]))
                    )
                point.set_features(
                    emade_data.FeatureData(feature_data)
                    )
                points.append(point)

            return (emade_data.EmadeData(points), None)

        # now mimic emade.GPFramework.EMADE.buildClassifier
        # https://github.gatech.edu/emade/emade/blob/CacheV2/src/GPFramework/EMADE.py#L343
        def reduce_instances(emadeDataTuple):
            '''
            wtf ...this method doesn't even use subset. gonna comment that ish out
            '''
            #proportion = self.datasetDict[dataset]['reduceInstances']
            emadeData, cache = emadeDataTuple
            #subset = emadeData.get_instances()[:round(len(emadeData.get_instances()) * proportion)]
            return emadeData, cache

        train_data_array = [reduce_instances(load_function(ezcgp_data_list[0]))]
        validate_data_array = [reduce_instances(load_function(ezcgp_data_list[1]))]
        test_data_array = [reduce_instances(load_function(ezcgp_data_list[2]))]

        # Copy the truth data in to its own location
        truth_data_array = [test_data[0].get_target() for test_data in validate_data_array]

        # Clear out the truth data from the test data
        [test_data[0].set_target(np.full(test_data[0].get_target().shape,np.nan)) for test_data in validate_data_array]

        # Stores DataPair object
        dataPairArray = [emade_data.EmadeDataPair(
                            train_data, test_data
                            ) for train_data, test_data in
                                zip(train_data_array, validate_data_array)]

        truthDataArray = truth_data_array

        import pdb; print("about to finish...let's check this out\n"); pdb.set_trace()
        
        return dataPairArray, truthDataArray, test_data_array



class ezDataLoader_EMADE_Titanic(ezDataLoader):
    '''
    going to try and mimic how EMADE loads in DataPairs
    for Titanic dataset

    assuming we used the emade.sh script and downloaded emade repo into datasets dir
    '''
    def __init__(self):
        super().__init__(1,0,0)


    def load(self, x, y):
        '''
        using the filepaths as it's read from the config xml in emade
        https://github.gatech.edu/emade/emade/blob/CacheV2/templates/input_titanic.xml
        '''
        train_datapair = ezData.ezData_EMADE(train_filenames=['datasets/titanic/train_0.csv.gz',
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

