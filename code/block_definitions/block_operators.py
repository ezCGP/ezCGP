'''
root/code/block_definitions/block_operators.py

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
sys.path.append(dirname(dirname(dirname(realpath(__file__))))) #one 'dirname' for every parentdir including root

### absolute imports wrt root
from code.block_definitions.utilities import tools
from code.block_definitions.utilities import operators_numpy # do something better...



class BlockOperators_Abstract():
    '''
    words
    '''
    def __init__(self,
                operators: List,
                weights: List,
                modules: List):
        self.build_operDict(modules)
        self.operators = operators
        self.weights = weights


    def build_operDict(self, modules: List):
        '''
        import the operator dict from every module in the list and return
        '''
        self.operator_dict = {}
        for oper_py in modules:
            _ = __import__(oper_py)
            self.operator_dict.update(_.operDict)
            del _


    def import_list(self):
        '''
        in theory import packages only if we use the respective EvaluateDefinition

        likely will abandon this...or should it be in operator definition?
        '''
        return []


    # TODO have a method to import in all modules and set weight_dict for all methods to 1



class BlockOperators_SymbRegressionOpsNoArgs(BlockOperators_Abstract):
    '''
    words
    '''
    def __init__(self):
        modules = ['simple_numpy']
        weight_dict = {simple_numpy.add_ff2f: 1,
                    simple_numpy.add_fa2a: 1,
                    simple_numpy.add_aa2a: 1,
                    simple_numpy.sub_ff2f: 1,
                    simple_numpy.sub_fa2a: 1,
                    simple_numpy.sub_aa2a: 1,
                    simple_numpy.mul_ff2f: 1,
                    simple_numpy.mul_fa2a: 1,
                    simple_numpy.mul_aa2a: 1}
        operators, weights = tools.build_weights(weight_dict)
        OperatorDefinition.__init__(self,
                                    operators,
                                    weights,
                                    modules)



class BlockOperators_SymbRegressionOpsWithArgs(BlockOperators_Abstract):
    '''
    words
    '''
    def __init__(self):
        modules = ['simple_numpy']
        weight_dict = {simple_numpy.add_aa2a: 1,
                    simple_numpy.sub_ff2f: 1,
                    simple_numpy.sub_fa2a: 1,
                    simple_numpy.sub_aa2a: 1,
                    simple_numpy.mul_ff2f: 1,
                    simple_numpy.mul_fa2a: 1,
                    simple_numpy.mul_aa2a: 1}
        operators, weights = tools.build_weights(weight_dict)
        OperatorDefinition.__init__(self,
                                    operators,
                                    weights,
                                    modules)