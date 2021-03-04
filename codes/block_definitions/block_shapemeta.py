'''
root/code/block_definitions/block_shapemeta.py

Overview:
I know, I know. It's a crap name. Basically this is a 'miscellaneous" attribute class. Provides zero methods, just input and output data types for the block, and number of nodes in a genome. Of course, the user can add any other attribute relevant to their problem and because of the for-loop in the __init__ method of BlockDefinition, it will add every single attribute here to the BlockDefinition.

Rules:
Only requirement is a list of input data types, list of ouptput data types, and an int for the number of nodes in a genome.
As mentioned above, the user can also add in any other attribute they want eventually added to BlockDefinition.
'''

### packages
import numpy as np

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from codes.utilities.custom_logging import ezLogging



class BlockShapeMeta_Abstract():
    '''
    Note, not an ABC.
    Only requires list of input dtypes, list of output dtypes, and main node count
    '''
    def __init__(self,
                 input_dtypes: list=[],
                 output_dtypes: list=[],
                 main_count: int=20,
                 **kwargs):
        ezLogging.debug("%s-%s - Initialize BlockShapeMeta_Abstract Class" % (None, None))
        self.input_dtypes = input_dtypes
        self.input_count = len(input_dtypes)
        self.output_dtypes = output_dtypes
        self.output_count = len(output_dtypes)
        self.main_count = main_count
        self.genome_count = self.input_count + self.output_count + self.main_count
        # add the misc kwargs
        for key, value in kwargs.items():
            self.__dict__[key] = value



class BlockShapeMeta_SymbolicRegressionNoArg25(BlockShapeMeta_Abstract):
    '''
    Reasonable block size to use for Symbolic Regression of 25 nodes.
    Note that an input ins a np.float64, so this will be used in the case where we do not want to use .args and rather use operators to evolve the input float into something useful for the regression.
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockShapeMeta_SymbolicRegression25 Class" % (None, None))
        input_dtypes = [np.float64, np.ndarray]
        output_dtypes = [np.ndarray]
        main_count = 25
        BlockShapeMeta_Abstract.__init__(self,
                                         input_dtypes,
                                         output_dtypes,
                                         main_count)



class BlockShapeMeta_SymbolicRegressionArg25(BlockShapeMeta_Abstract):
    '''
    Reasonable block size to use for Symbolic Regression of 25 nodes.
    Note that an input ins a np.float64, so this will be used in the case where we do not want to use .args and rather use operators to evolve the input float into something useful for the regression.
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockShapeMeta_SymbolicRegression25 Class" % (None, None))
        input_dtypes = [np.ndarray]
        output_dtypes = [np.ndarray]
        main_count = 25
        BlockShapeMeta_Abstract.__init__(self,
                                         input_dtypes,
                                         output_dtypes,
                                         main_count)



class BlockShapeMeta_Gaussian(BlockShapeMeta_Abstract):
    '''
    going to experiment with the size of the block relative to the number of gaussians needed to be fit in the data
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockShapeMeta_Gaussian Class" % (None, None))
        from misc import fake_mixturegauss
        input_dtypes = [fake_mixturegauss.XLocations, fake_mixturegauss.RollingSum]
        output_dtypes = [fake_mixturegauss.RollingSum]
        main_count = 50 #10 gaussians
        BlockShapeMeta_Abstract.__init__(self,
                                         input_dtypes,
                                         output_dtypes,
                                         main_count)



class BlockShapeMeta_DataAugmentation(BlockShapeMeta_Abstract):
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockShapeMeta_DataAugmentation Class" % (None, None))
        import Augmentor
        input_dtypes = [Augmentor.Pipeline]
        output_dtypes = [Augmentor.Pipeline]
        main_count = 10
        super().__init__(input_dtypes,
                         output_dtypes,
                         main_count)



class BlockShapeMeta_DataPreprocessing(BlockShapeMeta_Abstract):
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockShapeMeta_DataPreprocessing Class" % (None, None))
        import Augmentor
        input_dtypes = [Augmentor.Pipeline]
        output_dtypes = [Augmentor.Pipeline]
        main_count = 10
        super().__init__(input_dtypes,
                         output_dtypes,
                         main_count)



class BlockShapeMeta_Augmentor_TransferLearning(BlockShapeMeta_Abstract):
    '''
    Note that even though the models are type tf.keras.Models,
    we are adding them as 'Augmentor.Operations.Operation' so the
    input/output data types are Augmentor.Pipelines
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockShapeMeta_TransferLearning Class" % (None, None))
        import Augmentor
        input_dtypes = [Augmentor.Pipeline]
        output_dtypes = [Augmentor.Pipeline]
        main_count = 3
        super().__init__(input_dtypes,
                         output_dtypes,
                         main_count)



class BlockShapeMeta_TFKeras_TransferLearning(BlockShapeMeta_Abstract):
    '''
    Note that even though the models are type tf.keras.Models,
    we are adding them as 'Augmentor.Operations.Operation' so the
    input/output data types are Augmentor.Pipelines
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockShapeMeta_TransferLearning Class" % (None, None))
        # don't want it imported all the time so we didn't put it at the top of script
        import tensorflow as tf
        input_dtypes = [tf.keras.layers]
        output_dtypes = [tf.keras.layers]
        main_count = 1 #has to be one if using BlockEvaluate_TFKeras_TransferLearning2()
        super().__init__(input_dtypes,
                         output_dtypes,
                         main_count)



class BlockShapeMeta_TFKeras(BlockShapeMeta_Abstract):
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockShapeMeta_TFKeras Class" % (None, None))
        # don't want it imported all the time so we didn't put it at the top of script
        import tensorflow as tf
        input_dtypes = [tf.keras.layers]
        output_dtypes = [tf.keras.layers]
        main_count = 10
        super().__init__([tf.keras.layers],
                         [tf.keras.layers],
                         10)
        
        self.batch_size = 128
        self.epochs = 20

class BlockShapeMeta_SimGAN_Network(BlockShapeMeta_Abstract):
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockShapeMeta_SimGAN_Network Class" % (None, None))
        # don't want it imported all the time so we didn't put it at the top of script
        from codes.block_definitions.utilities.operators_pytorch import PyTorchLayerWrapper
        import torch
        # TODO:  this should have hyperparameters related to SimGANs, i.e. the stuff we kept in the simgan.xml
        input_dtypes = [PyTorchLayerWrapper, PyTorchLayerWrapper]
        output_dtypes = [PyTorchLayerWrapper]
        main_count = 30 # TODO: see if this needs to be bumped up
        super().__init__(input_dtypes,
                         output_dtypes,
                         main_count)
        
        ### CONFIG ###
        self.device = torch.device('cpu')

        # TODO: find a way to evolve these
        # TODO: find out if this is appropriate number. Is there a way to short circuit?
        self.train_steps = 200
        self.r_pretrain_steps = 10
        self.d_pretrain_steps = 10
        self.d_updates_per_train_step = 1
        self.r_updates_per_train_step = 2

        # Optim
        self.r_lr = 1e-4
        self.d_lr = 1e-4
        self.delta = 1e-4

        # Losses
        self.self_regularization_loss = torch.nn.L1Loss(reduction='sum')
        self.local_adversarial_loss = torch.nn.CrossEntropyLoss(reduction='mean')

        # Logging
        self.steps_per_log = 100

        # Using image history
        self.use_image_history = True