import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import os
import six
from six.moves import cPickle as pickle

from utils.training_block import TrainingBlock
from utils.preprocessing_block import PreprocessingBlock

from utils.DbConfig import DbConfig
from utils.DbManager import DbManager

# testing a small commit

generation_limit = 19
score_min = 0.00 # terminate immediately when 100% accuracy is achieved

db_config = DbConfig()
manager = DbManager(db_config)

#
#
def get_dummy_data(data_dimensions):
    # function that returns dummy datapoints and label so that evaluate
    # can properly build the tensorflow graph with knowledge of dimensions
    # adding [1] for #block inputs
    return (np.zeros([1] + data_dimensions), np.zeros(data_dimensions[0]))


train, test, val =  manager.load_CIFAR10()
x_train, y_train = train
x_test, y_test = test
x_val, y_val = val

# Invoke the above function to get our data.
# x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)
# TRAIN_SIZE = 1000
# # TRAIN_SIZE = x_train.shape[0]

# x_train = np.array([x_train[:TRAIN_SIZE]])
# y_train = np.array(y_train[:TRAIN_SIZE])

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
x_train, y_train = get_dummy_data([1000, 32, 32, 3])

def scoreFunction(predict, actual):
    try:
        acc_score = accuracy_score(actual, predict)
        avg_f1_score = f1_score(actual, predict, average='macro')
        return 1 - acc_score, 1 - avg_f1_score
    except ValueError:
        print('Malformed predictions passed in. Setting worst fitness')
        return 1, 1 # 0 acc_score and avg f1_score b/c we want this indiv ignored

"""
    Play with difference sizes, and different distribution

    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print(x_train.shape[0])
    # val_size = int(0.1 * x_train.shape[0]) # percentage of training data
    val_size = 2000 # exact value done so that x_train has a size multiple of batch_size
    print(val_size)
    val_ind = np.random.choice(a=np.arange(x_train.shape[0]), size=val_size, \
        replace=False)
    val_mask = np.zeros(x_train.shape[0], dtype=bool)
    val_mask[val_ind] = True

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    x_val = x_train[val_mask]
    y_val = y_train[val_mask]

    x_train = x_train[~val_mask]
    y_train = y_train[~val_mask]

    x_train = [x_train]
    x_test = [x_test]


    print('Train: X: {} y: {}'.format(x_train[0].shape, y_train.shape))
    print('Validation: X: {} y: {}'.format(x_val.shape, y_val.shape))
    print('Test: X: {} y: {}'.format(x_test[0].shape, y_test.shape))

    print('Loaded MNIST dataset. x_train: {} y_train: {} x_test: {} y_test: {}'
        .format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

"""
preprocessing_block = PreprocessingBlock()
#print('preprocessing block: ', vars(preprocessing_block))

training_block = TrainingBlock(main_count=30,learning_required=True)
#print('training block: ', vars(training_block))

skeleton_genome = { # this defines the WHOLE GENOME
    'input': [np.ndarray], # we don't pass in the labels since the labels are only used at evaluation and scoring time
    'output': [np.ndarray],
    # vars converts the block object to a dictionary
    1: vars(preprocessing_block),
    2: vars(training_block)
}

print("Data X, Y:")
print(x_test)
print(y_test)
