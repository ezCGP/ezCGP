'''
root/code/block_definitions/block_shape.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
import numpy as np

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__))))) #one 'dirname' for every parentdir including root

### absolute imports wrt root



class BlockShapeMeta_Abstract():
    '''
    a lot of this is just to help fill attributes of a block
    like number of nodes, acceptable input/output datatypes, etc
    '''
    def __init__(self,
                 input_dtypes: list=[],
                 output_dtypes: list=[],
                 main_count: int=20):
        self.input_dtypes = input_dtypes
        self.input_count = len(input_dtypes)
        self.output_dtypes = output_dtypes
        self.output_count = len(output_dtypes)
        self.main_count = main_count
        self.genome_count = self.input_count + self.output_count + self.main_count



class BlockShapeMeta_SymbolicRegression25(BlockShapeMeta_Abstract):
	'''
	TODO
	'''
    def __init__(self):
        input_dtypes = [np.float64, np.ndarray]
        output_dtypes = [np.ndarray]
        main_count = 25
        BlockShapeMeta_Abstract.__init__(self,
		                                 input_dtypes,
		                                 output_dtypes,
		                                 main_count)