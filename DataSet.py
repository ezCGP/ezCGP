import numpy as np


class DataSet():
    def __init__(self, x, y):
        super().__init__()
        self.__index_in_epoch = 0
        self._index_in_epoch = 0
        self._num_examples = len(x)
        self._epochs_completed = 0
        self._features = x
        self._labels = y

    def clear_batch(self):
        self.__index_in_epoch = 0
        self._index_in_epoch = 0
        self._num_examples = 0
        self._epochs_completed = 0
        self._features = []
        self._labels = []

    def next_batch(self, batch_size: int = 128):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            assert batch_size <= self._num_examples
            if self._index_in_epoch - batch_size == self._num_examples:
                start = 0
                self._index_in_epoch = batch_size
            else:
                ret_image, ret_label = self._features[self._index_in_epoch - batch_size:], \
                                       self._labels[self._index_in_epoch - batch_size:]
                self._index_in_epoch = 0
                return ret_image, ret_label
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]

    def x_y_list(self) -> (list, list):
        return list(self._features), list(self._labels)
