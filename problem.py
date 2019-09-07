# TODO edit problem towards mnist usage

import numpy as np
#from scipy.stats import weibull_min
import scipy.stats as scst
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
import os
import six
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

import operators
import arguments
import mutate_methods as mut
import mate_methods as mate
from DbConfig import DbConfig
from DbManager import DbManager

# constants
generation_limit = 19
score_min = 0.00 # terminate immediately when 100% accuracy is achieved


# Helper to load in CIFAR-10
def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        if six.PY2:
            datadict = pickle.load(f)
        elif six.PY3:
            datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = './cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)

    # X_train -= mean_image
    # X_val -= mean_image
    # X_test -= mean_image

    # Reshape data to rows
    # X_train = X_train.reshape(num_training, -1)
    # X_val = X_val.reshape(num_validation, -1)
    # X_test = X_test.reshape(num_test, -1)

    return X_train/255.0, y_train, X_val/255.0, y_val, X_test/255.0, y_test

def get_cifar10_batch(filename):
    with open(filename, 'rb') as f:
        if six.PY2:
            datadict = pickle.load(f)
        elif six.PY3:
            datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y


def get_dummy_data(data_dimensions):
    # function that returns dummy datapoints and label so that evaluate
    # can properly build the tensorflow graph with knowledge of dimensions
    # adding [1] for #block inputs
    return (np.zeros([1] + data_dimensions), np.zeros(data_dimensions[0]))

# Invoke the above function to get our data.
x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)
TRAIN_SIZE = 1000
# TRAIN_SIZE = x_train.shape[0]

x_train = np.array([x_train[:TRAIN_SIZE]])
y_train = np.array(y_train[:TRAIN_SIZE])

x_test = np.array([x_test])

print('Train data shape: ', x_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', x_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)

# we make dummy data that is dimensionally representative of the data we will
# feed in to the individual so that the evaluate method can accurately construct
# the tensorflow graph
x_train, y_train = get_dummy_data([1, 32, 32, 3])

def scoreFunction(predict, actual):
    try:
        acc_score = accuracy_score(actual, predict)
        avg_f1_score = f1_score(actual, predict, average='macro')
        return 1 - acc_score, 1 - avg_f1_score
    except ValueError:
        print('Malformed predictions passed in. Setting worst fitness')
        return 1, 1 # 0 acc_score and avg f1_score b/c we want this indiv ignored

# play with difference sizes, and different distribution

# mnist = tf.keras.datasets.mnist
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0


# print(x_train.shape[0])
# # val_size = int(0.1 * x_train.shape[0]) # percentage of training data
# val_size = 2000 # exact value done so that x_train has a size multiple of batch_size
# print(val_size)
# val_ind = np.random.choice(a=np.arange(x_train.shape[0]), size=val_size, \
#     replace=False)
# val_mask = np.zeros(x_train.shape[0], dtype=bool)
# val_mask[val_ind] = True

# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)

# x_val = x_train[val_mask]
# y_val = y_train[val_mask]

# x_train = x_train[~val_mask]
# y_train = y_train[~val_mask]

# x_train = [x_train]
# x_test = [x_test]



# print('Train: X: {} y: {}'.format(x_train[0].shape, y_train.shape))
# print('Validation: X: {} y: {}'.format(x_val.shape, y_val.shape))
# print('Test: X: {} y: {}'.format(x_test[0].shape, y_test.shape))

# print('Loaded MNIST dataset. x_train: {} y_train: {} x_test: {} y_test: {}'
#     .format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

# NOTE: a lot of this is hastily developed and I do hope to improve the 'initialization'
#structure of the genome; please note your own ideas and we'll make that a 'project' on github soon


skeleton_block = { #this skeleton defines a SINGLE BLOCK of a genome
    'tensorblock_flag': True, # flag if it's input or a training block
    'batch_size': 128, # split one image into group of 128 pixels
    'n_epochs': 1, #the number of epochs to run for while training
    # 'large_dataset': None,
    'large_dataset': (['cifar-10-batches-py/data_batch_1', \
        'cifar-10-batches-py/data_batch_2', \
        'cifar-10-batches-py/data_batch_3'], get_cifar10_batch),
    'nickname': 'tensor_mnist_block',
    'setup_dict_ftn': {
        #declare which primitives are available to the genome,
        #and assign a 'prob' so that you can control how likely a primitive will be used;
        #prob: float btwn 0 and 1 -> assigns that prob to that primitive...the sum can't be more than 1
        #prob: 1 -> equally distribute the remaining probability amoung all those remaining (hard to explain, sorry)
        #operators.input_layer: {'prob': 1},
        #operators.add_tensors: {'prob': 1},
        #operators.sub_tensors: {'prob': 1},
        #operators.mult_tensors: {'prob': 1},
        operators.dense_layer: {'prob': 1},
        operators.conv_layer: {'prob': 1},
        operators.max_pool_layer: {'prob': 1},
        # operators.avg_pool_layer: {'prob': 1},
        # operators.concat_func: {'prob': 1},
        # operators.sum_func: {'prob': 1},
        # operators.conv_block: {'prob': 1},
        # operators.res_block: {'prob': 1},
        #operators.sqeeze_excitation_block: {'prob': 1},
        #operators.identity_block: {'prob': 1}, # TODO replace this with info from operator_dict?
    },
    'setup_dict_arg': {
        #if you have an 'arguments genome', declare which argument-datatypes should fill the argument genome
        #not used for now...arguments genome still needs to be tested
        # arguments.argInt: {'prob': 1}}, #arg count set to 0 though
        arguments.argPow2: {'prob': 1},
        arguments.argFilterSize: {'prob': 1}},
    'setup_dict_mate': {
        #declare which mating methods are available to genomes
        mate.Mate.dont_mate: {'prob': 1, 'args': []}},
    'setup_dict_mut': {
        #declare which mutation methods are available to the genomes
        mut.Mutate.mutate_singleInput: {'prob': 1, 'args': []},
        mut.Mutate.mutate_singleArgValue: {'prob': 1, 'args': []},
        mut.Mutate.mutate_singleArgIndex: {'prob': 1, 'args': []},
        mut.Mutate.mutate_singleFtn: {'prob': 1, 'args': []},
    },
    'operator_dict': operators.operDict, #further defines what datatypes what arguments are required for each primitive
    'block_input_dtypes': [tf.Tensor], #placeholder datatypes so that the genome can be built off datatypes instead of real data
    'block_outputs_dtypes': [tf.Tensor],
    'block_main_count': 4, #10 genes
    'block_arg_count': 20, #not used...no primitives require arguments
    'block_mut_prob': 1, #mutate genome with probability 1...always
    'block_mate_prob': 0 #mate with probability 0...never
}

skeleton_genome = { # this defines the WHOLE GENOME
    'input': [np.ndarray], # we don't pass in the labels since the labels are only used at evaluation and scoring time
    'output': [np.ndarray],
    1: skeleton_block
}

print("Data X, Y:")
print(x_test)
print(y_test)