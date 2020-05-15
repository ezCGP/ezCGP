'''
root/code/block_definitions/block_arguments.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
from typing import List
import inspect

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
    def __init__(self):
        self.arg_count = 0
        self.each_type = []
        self.each_weight = []
        self.arg_types = []


    def init_from_weight_dict(self, weight_dict):
        '''
        TODO
        '''
        args, weights = tools.build_weights(weight_dict)
        self.each_type = args
        self.each_weight = weights
        self.set_arg_types()


    def get_all_classes(self, module='argument_types'):
        vals = inspect.getmembers(globals()[module], inspect.isclass)
        # vals will be a list of tuples (name, value)...we want the value
        all_classes = []
        for name, value in vals:
            all_classes.append(value)

        return all_classes


    def set_equal_weights(self, module='argument_types'):
        weight_dict = {}
        for arg_type in self.get_all_classes(module):
            weight_dict[arg_type] = 1

        return weight_dict


    def set_arg_types(self):
        '''
        note it only fills it by the data type class not instances of the arg class
        '''
        start_point = 0
        end_point = 0
        self.arg_types = [None]*self.arg_count
        for arg_class, arg_weight in zip(self.each_type, self.each_weight):
            end_point += int(arg_weight*self.arg_count)
            for arg_index in range(start_point, end_point):
                self.arg_types[arg_index] = arg_type
            start_point = end_point

        if end_point != self.arg_count:
            # prob some rounding errors then
            sorted_byweight = np.argsort(self.each_weight)[::-1] # sort then reverse to go from largest to smallest
            for i, arg_index in enumerate(range(end_point, self.arg_count)):
                arg_class = self.each_type[sorted_byweight[i]]
                self.arg_types[arg_indx] = arg_class
        else:
            pass



class BlockArgumentsSize50(BlockArguments_Abstract):
    def __init__(self):
        BlockArguments_Abstract.__init__(self)
        self.arg_count = 50
        arg_dict = {argument_types.ArgumentType_Ints: 1,
                    argument_types.ArgumentType_Pow2: 1}
        self.init_from_weight_dict(arg_dict)



class BlockArgumentsNoArgs(BlockArguments_Abstract):
    def __init__(self):
        BlockArguments_Abstract.__init__(self)