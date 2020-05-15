'''
root/code/block_definitions/block_operators.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

get list of operator scripts/modules to import + set the weights for the operators

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
from typing import List
from copy import deepcopy
import importlib
import inspect

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from codes.block_definitions.utilities import tools



class BlockOperators_Abstract():
    '''
    words
    mention about operators and weights list of same len
    '''
    def __init__(self):
        self.operator_dict = {}
        self.operators = []
        self.weights = []


    def init_from_weight_dict(self, weight_dict):
        operators, weights = tools.build_weights(weight_dict)
        self.operators = operators
        self.weights = weights


    def import_operator_scripts(self, module_names, module_aliases=None):
        '''
        vars() returns a dictionary of local variables...starts out empty inside a function

        vars() -> locals()
        want to eventually change globals()

        note that sys.path is a 'global' variable so the paths added earlier in the file can be used here
        '''
        if module_aliases is None:
            module_aliases = deepcopy(module_names)
        else:
            assert(len(module_aliases) == len(module_names)), "module names and aliases need to be the same length"

        for name, alias in zip(module_names, module_aliases):
            #globals()[alias] = __import__(name)
            #going to use importlib.import_module instead of __import __ because of convention and to do better absolute/relative imports
            globals()[alias] = importlib.import_module("codes.block_definitions.utilities.%s" % name)
            self.operator_dict.update(globals()[alias].operator_dict)


    def get_all_functions(self, module):
        vals = inspect.getmembers(globals()[module], inspect.isfunction)
        # vals will be a list of tuples (name, value)...we want the value
        all_functions = []
        for name, value in vals:
            all_functions.append(value)

        return all_functions


    def set_equal_weights(self, module):
        weight_dict = {}
        for func in self.get_all_functions(module):
            weight_dict[func] = 1

        return weight_dict



class BlockOperators_SymbRegressionOpsNoArgs(BlockOperators_Abstract):
    '''
    words
    '''
    def __init__(self):
        BlockOperators_Abstract.__init__(self)

        modules = ['operators_symbregression_noargs']
        self.import_operator_scripts(modules)

        weight_dict = {}
        for module in modules:
            weight_dict.update(self.set_equal_weights(module))

        self.init_from_weight_dict(weight_dict)



class BlockOperators_SymbRegressionOpsWithArgs(BlockOperators_Abstract):
    '''
    words
    '''
    def __init__(self):
        BlockOperators_Abstract.__init__(self)

        modules = ['operators_symbregression_args']
        self.import_operator_scripts(modules)

        weight_dict = {}
        for module in modules:
            weight_dict.update(self.set_equal_weights(module))

        self.init_from_weight_dict(weight_dict)