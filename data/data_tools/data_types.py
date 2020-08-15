"""
Created by Michael Jurado and Yuhang Li.
This class will be how ezCGP transfers data between blocks.
"""

from PIL import Image
import Augmentor
import numpy as np

class ezDataSet():

    def __init__(self, x_train, y_train, x_test=None, y_test=None):
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
        self.augmentation_pipeline = Augmentor.Pipeline()
        self.preprocess_pipeline = Augmentor.Pipeline()
        self.batch_size = None  # batch_size
        self.generator = None  # training_generator


    def _transform_preprocess(self, images):
        """
        purpose of function is to mirror Augmentor.Pipelines torch_transform method for preprocessing

         - Recall that preprocessing methods should have a probability of 1, which is why this function errors if
           any primtives have a probability less than 1

          - We don't call torch_transform here it is inefficient
        images: all images (like x_train)
        """
        for operation in self.preprocess_pipeline.operations: # assure that all probabilities are set to 1
            assert operation.probability == 1
        for operation in self.preprocess_pipeline.operations:
            images = operation.perform_operation(images)
        return images

    def clear_data(self):
        """
        Method clears the data structures so that individual can be re-evaluated
        """
        self.batch_size = None
        self.generator = None
        self.augmentation_pipeline = Augmentor.Pipeline()
        self.preprocess_pipeline = Augmentor.Pipeline()

    def next_batch_train(self, batch_size):
        """
        Applies augmentation and preprocessing pipeline to a batch of data
            - Use this in evaluator methods to train a network
        batch_size: amount of data to sample for x_train, and y_train
        returns: x_batch, y_batch -> len(x_batch) = batch_size
        """
        # augmentation pipeline
        images = np.random.choice(np.arange(len(self.x_train)),  batch_size)
        x_batch = self.x_train[images]
        x_batch = [Image.fromarray(img) for img in x_batch]
        y_batch = self.y_train[images]
        augmentation = self.augmentation_pipeline.torch_transform() # applies augmentation according to probabilities
        x_batch = np.array([np.asarray(augmentation(x)) for x in x_batch])

        # preprocess pipeline
        return np.asarray(self._transform_preprocess(x_batch)), y_batch

    def make_generator(self, batch_size):
        """
        DEPRECATED because it does not apply the preprocessing pipeline correctly
        Makes a keras data_generator from the train pipeline
        param batch_size: mini-batch size
        :return None
        """
        if batch_size != self.batch_size:
            self.generator = self.augmentation_pipeline.keras_generator_from_array(self.x_train.astype(np.uint8),
                                                                            self.y_train, batch_size, scaled=False)

    def preprocess_train_data(self):
        """
        Runs the reprocessing pipeline and returns  preprocessed train data
        :return: preprocessed train data (unbatched)
        """
        return np.asarray(self._transform_preprocess(self.x_train)), self.y_train

    def preprocess_test_data(self):
        """
        Runs the preprocessing pipeline and returns preprocessed test data
        :return: preprocessed test data (unbatched)
        """
        return np.asarray(self._transform_preprocess(self.x_test)), self.y_test


class ezDataSet_BatchImages(ezDataSet):
    pass