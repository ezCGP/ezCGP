"""
Helper Methods should go here make sure to add comments
File Created by Michael Jurado, Trai Tan, and Mai Pham"""
import numpy as np
import six
from six.moves import cPickle as pickle



"""Datasets we Can Load:
    -Cifar-10 with xshape (10000, 3, 32, 32)"""
def load_data_from_path(path, xshape):
    """
    path: location of cifar 10 data
    xshape: desired shape of x. IE (Samples, Channels, Width, Height)
    returns X, y -> training data pair
    """
    with open(path, 'rb') as f:
        if six.PY2:
            datadict = pickle.load(f)
        elif six.PY3:
            datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(*xshape).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

"""Useful for standard machine learning where order does not matter and data is randomly pre-shuffled."""
def split_data(X, y, train_size_perc, validation_size_perc):
    assert train_size_perc + validation_size_perc == 1

    train_index = int(len(X) * train_size_perc)
    X_train = X[0:train_index]
    y_train = y[0:train_index]

    X_val = X[train_index:]
    y_val = y[train_index:]

    return (X_train, y_train), (X_val, y_val)
