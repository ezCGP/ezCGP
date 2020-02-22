"""
Created by Michael Jurado with help from William.
This class will be how ezCGP transfers training data between blocks
"""
import numpy as np
import Augmentor
from Augmentor.Operations import Operation

class DataSet():
    def __init__(self, x_train, y_train, x_test, y_test):
        """

        :param x_train: training samples
        :param y_train: training labels
        :param x_test: testing samples
        :param y_test: testing labels
        """
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.train_pipeline = Augmentor.Pipeline()
        self.test_pipeline = Augmentor.Pipeline()
        self.batch_size = None  # batch_size
        self.generator = None  # training_generator

    def clear_data(self):
        """
        Method clears the data structures so that individual can be re-evaluated
        :return:
        """
        self.batch_size = None

        del(self.generator)
        self.generator = None
        
        del(self.train_pipeline)
        self.train_pipeline = Augmentor.Pipeline()

        del(self.test_pipeline)
        self.test_pipeline = Augmentor.Pipeline()

    def next_batch_train(self, batch_size):
        """

        :param batch_size: mini-batch size
        :return: numpy array of training samples
        """
        if batch_size != self.batch_size:
            del(self.generator)
            self.generator = self.train_pipeline.keras_generator_from_array(self.x_train.astype(np.uint8),
                                                                            self.y_train, batch_size, scaled=False)
        return next(self.generator)

    def preprocess_data(self):
        """
        Runs the preprocessing pipeline and returns preprocessed testing data
        :return: preprocessed data (unbatched)
        """
        preMethod = self.test_pipeline.torch_transform()
        x_val_norm = [np.asarray(preMethod(x)) for x in self.x_test]
        for i in self.x_test:
            new = preMethod(i)
        return np.array(x_val_norm), self.y_test