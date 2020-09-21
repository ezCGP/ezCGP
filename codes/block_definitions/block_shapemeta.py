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
from data.data_tools import data_types
from codes.utilities.custom_logging import ezLogging



class BlockShapeMeta_Abstract():
    '''
    Note, not an ABC.
    Only requires list of input dtypes, list of output dtypes, and main node count
    '''
    def __init__(self,
                 input_dtypes: list=[],
                 output_dtypes: list=[],
                 main_count: int=20):
        ezLogging.debug("%s-%s - Initialize BlockShapeMeta_Abstract Class" % (None, None))
        self.input_dtypes = input_dtypes
        self.input_count = len(input_dtypes)
        self.output_dtypes = output_dtypes
        self.output_count = len(output_dtypes)
        self.main_count = main_count
        self.genome_count = self.input_count + self.output_count + self.main_count



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

class BlockShapeMeta_Keras(BlockShapeMeta_Abstract):
    def __int__(self):
        ezLogging.debug("%s-%s - Initialize BlockShapeMeta_Keras Class" % (None, None))
        import tensorflow as tf
        input_dtypes = [tf.keras.layers]
        output_dtypes = [tf.keras.layers]
        main_count = 25
        BlockShapeMeta_Abstract.__init__(self,
                                     input_dtypes,
                                     output_dtypes,
                                     main_count)

