'''
root/code/block_definitions/block_arguments.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
from typing import List

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from codes.block_definitions.utilities import tools
from codes.block_definitions.utilities import argument_types



class BlockArguments_Abstract():
    '''
    words
    '''
    def __init__(self,
                arg_count: int,
                arg_types: List,
                arg_weights: List):
        self.arg_count = arg_count
        self.arg_alltypes = arg_types
        self.arg_weights = arg_weights
        self.fill_args()


    def get_arg_weights(weight_dict):
        '''
        only works before __init__ called
        '''
        args, weights = build_weights(weight_dict)


    def fill_args(self):
        '''
        note it only fills it by the data type class not instances of the argtype
        '''
        start_point = 0
        end_point = 0
        self.arg_types = [None]*self.arg_count
        for arg_type, arg_weights in zip(self.arg_alltypes, self.arg_weights):
            end_point += int(arg_weight*self.arg_count)
            for arg_index in range(start_point, end_point):
                self.arg_types[arg_index] = arg_type
            start_point = end_point
        if end_point != self.arg_count:
            # prob some rounding errors then
            sorted_byweight = np.argsort(self.arg_weights)[::-1] # sort then reverse
            for i, arg_index in enumerate(range(end_point, self.arg_count)):
                arg_class = self.arg_alltypes[sorted_byweight[i]]
                self.arg_types[arg_indx] = arg_class
        else:
            pass



class BlockArgumentsSize50(BlockArguments_Abstract):
    def __init__(self):
        arg_count = 50
        arg_dict = {argument_types.ArgumentType_Ints: 1,
                    argument_types.ArgumentType_Pow2: 1}
        args, weights = ArgumentDefinition.get_arg_weights(arg_dict)
        ArgumentDefinition.__init__(self,
                                    arg_count,
                                    args,
                                    weights)



class BlockArgumentsNoArgs(BlockArguments_Abstract):
    def __init__(self):
        arg_count = 0 
        ArgumentDefinition.__init__(self,
                                    arg_count,
                                    [],
                                    [])