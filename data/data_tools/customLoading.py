from data.data_tools.data_types import ezDataSet

import numpy as np

x_train = np.zeros((10000, 3, 128, 128), dtype = np.uint8)

y_train = np.ones((10000),dtype = np.uint8)

x_test = np.zeros((5000, 3, 128, 128),dtype = np.uint8)

y_test = np.ones((5000),dtype = np.uint8)

dataset = ezDataSet(x_train, y_train, x_test, y_test)

x_batch, y_batch = dataset.next_batch_train(100)
