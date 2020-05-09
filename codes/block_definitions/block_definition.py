'''
root/code/block/block_definition.py

Overview:
defined by shape/meta data, mate methods,mutate methods, evaluate method, operators, or primitives, argument datatypes

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
from numpy import random as rnd
import numpy as np

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from codes.block_definitions.block_shapemeta import BlockShapeMeta_Abstract
from codes.block_definitions.block_operators import BlockOperators_Abstract
from codes.block_definitions.block_arguments import BlockArguments_Abstract
from codes.block_definitions.block_evaluate import BlockEvaluate_Abstract
from codes.block_definitions.block_mutate import BlockMutate_Abstract
from codes.block_definitions.block_mate import BlockMate_Abstract
from codes.genetic_material import IndividualMaterial, BlockMaterial
from data.data_tools.data_types import ezDataSet



class BlockDefinition():
    '''
    TODO
    '''
    def __init__(self,
                 nickname: str,
                 meta_def: BlockShapeMeta_Abstract,
                 operator_def: BlockOperators_Abstract,
                 argument_def: BlockArguments_Abstract,
                 evaluate_def: BlockEvaluate_Abstract,
                 mutate_def: BlockMutate_Abstract,
                 mate_def: BlockMate_Abstract):
        # Meta:
        self.nickname = nickname
        self.meta_def = meta_def()
        for name, val in self.meta_def.__dict__.items():
            # quick way to take all attributes and add to self
            self.__dict__[name] = val
        # Mutate:
        self.mutate_def = mutate_def()
        self.prob_mutate = self.mutate_def.prob_mutate
        self.num_mutants = self.mutate_def.num_mutants
        # Mate:
        self.mate_def = mate_def()
        self.prob_mate = self.mate_def.prob_mate
        # Evaluate:
        self.evaluate_def = evaluate_def()
        # Operator:
        self.operator_def = operator_def()
        self.operator_dict = self.operator_def.operator_dict
        self.operator_dict["input"] = self.meta_def.input_dtypes
        self.operator_dict["output"] = self.meta_def.output_dtypes
        self.operators = self.operator_def.operators
        self.operator_weights = self.operator_def.weights
        # Argument:
        self.argument_def = argument_def()
        self.arg_count = self.argument_def.arg_count
        self.arg_types = self.argument_def.arg_types


    def init_block(self, block_material: BlockMaterial):
        '''
        define:
         * block_material.genome
         * block_material.args
         * block_material.need_evaluate
        '''
        block_material.need_evaluate = True
        self.fill_args(block_material)
        self.fill_genome(block_material)
        self.get_actives(block_material)


    def get_node_dtype(self, block_material: BlockMaterial, node_index: int, key: str):
        '''
        key returns that key-value from the respective node_dictionary
         * "inputs"
         * "args"
         * "output"
        '''
        if node_index < 0:
            # input_node
            return self.input_dtypes[-1*node_index-1]
        elif node_index >= self.main_count:
            # output_node
            return self.output_dtypes[node_index-self.main_count]
        else:
            # main_node
            node_ftn = block_material[node_index]["ftn"]
            oper_dict_value = self.operator_dict[node_ftn]
            return oper_dict_value[key]


    def get_random_input(self, block_material: BlockMaterial, req_dtype, _min=None, _max=None, exclude=[]):
        '''
        note _max is exclusive so [_min,_max)

        return None if we failed to find good input
        '''
        if _min is None:
            _min = -1*self.input_count
        if _max is None:
            _max = self.main_count
        
        choices = np.arange(_min, _max)
        for val in exclude:
            choices = np.delete(choices, np.where(choices==val))

        if len(choices) == 0:
            # nothing left to choose from
            return None
        else:
            # exhuastively try each choice to see if we can get datatypes to match
            poss_inputs = np.random.choice(a=choices, size=len(choices), replace=False)
            for input_index in poss_inputs:
                input_dtype = self.get_node_dtype(block_material, input_index, "output")
                if req_dtype == input_dtype:
                    return input_index
                else:
                    pass
            # none of the poss_inputs worked, failed to find matching input
            return None


    def get_random_ftn(self, req_dtype=None, exclude=[], return_all=False):
        '''
        TODO
        '''
        choices = np.array(self.operators)
        weights = np.array(self.operator_weights)
        
        for val in exclude:
            #note: have to delete from weights first because we use choices to get right index
            weights = np.delete(weights, np.where(choices==val))
            choices = np.delete(choices, np.where(choices==val))
        
        # now check the output dtypes match
        if req_dtype is not None:
            delete = []
            for ith_choice, choice in enumerate(choices):
                if self.operator_dict[choice]["output"] != req_dtype:
                    delete.append(ith_choice)
            weights = np.delete(weights, delete)
            choices = np.delete(choices, delete)

        if weights.sum() < 1 - 1e-3:
            # we must have removed some values...normalize
            weights *= 1/weights.sum()

        if return_all:
            return rnd.choice(choices, size=len(choices), replace=False, p=weights)
        else:
            return rnd.choice(choices, p=weights)


    def get_random_arg(self, req_dtype, exclude=[]):
        '''
        TODO
        '''
        choices = []
        for arg_index, arg_type in enumerate(self.arg_types):
            if (arg_type == req_dtype) and (arg_index not in exclude):
                choices.append(arg_index)

        if len(choices) == 0:
            return None
        else:
            return rnd.choice(choices)


    def fill_args(self, block_material: BlockMaterial):
        '''
        TODO
        '''
        block_material.args = [None]*self.arg_count
        for arg_index, arg_type in enumerate(self.arg_types):
            block_material.args[arg_index] = arg_type()


    def fill_genome(self, block_material: BlockMaterial):
        '''
        TODO
        '''
        block_material.genome = [None]*self.genome_count
        block_material.genome[(-1*self.input_count):] = ["InputPlaceholder"]*self.input_count

        # fill main nodes
        for node_index in range(self.main_count):
            ftns = self.get_random_ftn(return_all=True)
            for ftn in ftns:
                # find inputs
                input_dtypes = self.operator_dict[ftn]["inputs"]
                input_index = [None]*len(input_dtypes)
                for ith_input, input_dtype in enumerate(input_dtypes):
                    input_index[ith_input] = self.get_random_input(block_material, req_dtype=input_dtype, _max=node_index)
                if None in input_index:
                    # failed to fill it in; try another ftn
                    continue
                else:
                    pass

                # find args
                arg_dtypes = self.operator_dict[ftn]["args"]
                arg_index = [None]*len(arg_dtypes)
                for ith_arg, arg_dtype in enumerate(arg_dtypes):
                    poss_arg_index = self.get_random_arg(req_dtype=arg_dtype)
                if None in arg_index:
                    # failed to fill it in; try another ftn
                    continue
                else:
                    pass

                # all complete
                block_material[node_index] = {"ftn": ftn,
                                    "inputs": input_index,
                                    "args": arg_index}
                break
            # error check that node got filled
            if block_material[node_index] is None:
                print("GENOME ERROR: no primitive was able to fit into current genome arrangment")
                exit()

        # fill output nodes
        for ith_output, node_index in enumerate(range(self.main_count, self.main_count+self.output_count)):
            req_dtype = self.output_dtypes[ith_output]
            block_material[node_index] = self.get_random_input(block_material, req_dtype=req_dtype)


    def get_actives(self, block_material: BlockMaterial):
        '''
        TODO
        '''
        block_material.active_nodes = set(np.arange(self.main_count, self.main_count+self.output_count))
        block_material.active_args = set()
        #block_material.active_ftns = set()

        # add feeds into the output_nodes
        for node_input in range(self.main_count, self.main_count+self.output_count):
            block_material.active_nodes.update([block_material[node_input]])

        for node_index in reversed(range(self.main_count)):
            if node_index in block_material.active_nodes:
                # then add the input nodes to active list
                block_material.active_nodes.update(block_material[node_index]["inputs"])
                block_material.active_args.update(block_material[node_index]["args"])
                '''# if we need to check for learners...
                if (not block_material.has_learner) and ("learner" in block_material[node_index]["ftn"].__name__):
                    block_material.has_learner = True
                else:
                    pass'''
            else:
                pass

        # sort
        block_material.active_nodes = sorted(list(block_material.active_nodes))
        block_material.active_args = sorted(list(block_material.active_args))


    def mutate(self, indiv_material: IndividualMaterial, block_index: int):
        '''
        TODO
        '''
        self.mutate_def.mutate(indiv_material, block_index, self)
        self.get_actives(indiv_material[block_index])


    def mate(self, parent1: IndividualMaterial, parent2: IndividualMaterial, block_index: int):
        '''
        TODO
        '''
        #children: List() = self.mate_def.mate(parent1, parent2, block_index)
        children = self.mate_def.mate(parent1, parent2, block_index, self)
        for child in children:
            self.get_actives(child[block_index])
        return children


    def evaluate(self, block_material: BlockMaterial, training_datapair: ezDataSet, validation_datapair=None):
        '''
        TODO
        '''
        # verify that the input data matches the expected datatypes
        # TODO make a rule that training_datapair always has to be a list??? would be easiest for code
        for input_dtype, input_data in zip(self.input_dtypes, training_datapair):
            if input_dtype != type(input_data):
                print("ERROR: datatypes don't match", type(input_data), input_dtype) # add a proper message here
                return

        output = self.evaluate_def.evaluate(self, block_material, training_datapair, validation_datapair)
        return output
