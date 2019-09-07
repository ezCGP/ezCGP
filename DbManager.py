import numpy as np
# from scipy.stats import weibull_min
import scipy.stats as scst
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
import os
import six
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import DataSet
from copy import deepcopy

import operators
import arguments
import mutate_methods as mut
import mate_methods as mate
import DbConfig


class DbManager():
    def __init__(self, config: DbConfig):
        self.train_data_set: DataSet = None
        self.test_data_set: DataSet = None
        self.db_conf: DbConfig = config
        pass

    def load_CIFAR_batch(filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            if six.PY2:
                datadict = pickle.load(f)
            elif six.PY3:
                datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
            return X, Y

    def load_CIFAR10(self):
        """ load all of cifar """
        path = './cifar-10-batches-py'
        data = []
        for b in range(1, 6):
            f = os.path.join(path, 'data_batch_%d' % (b,))
            data.append(self.load_data_from_path(f))
        x = np.concatenate([x[0] for x in data])
        y = np.concatenate((x[1] for x in data))
        train, test, val = self.split_data(x, y, config=self.db_conf)
        return train, test

    def load_cifar100(self):
        pass

    def load_data_from_path(self, path) -> DataSet:
        with open(path, 'rb') as f:
            if six.PY2:
                datadict = pickle.load(f)
            elif six.PY3:
                datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
            return X, Y

    def split_data(self, x, y, config: DbConfig) -> (DataSet, DataSet, DataSet):
        pass

    def get_train_data(self):
        return self.train_data_set

    def get_test_data(self):
        return self.test_data_set

    def clone_train_data(self):
        return deepcopy(self.train_data_set)

    def clone_test_data(self):
        return deepcopy(self.test_data_set)
