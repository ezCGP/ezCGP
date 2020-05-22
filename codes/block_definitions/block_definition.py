'''
root/code/block/block_definition.py

Overview:
A block, as uzed in ezCGP, is characterized in 6 ways:
    1. 'Operator' Definition: what operators (primitives or methods) are available to populate the genome
    2. 'Argument' Definition: what argument data types are available to the genome
    3. 'Evaluate' Definition: what should we do with the genome
    4. 'Mutate' Definition: how can we mutate the block's genome
    5. 'Mate' Definition: how can we mate the block's genome with other blocks
    6. 'ShapeMeta' Definition: how many genes in a genome; input and output data types; etc
    
This block definition class is sort of a 'wrapper' class to join together the 6 definitions as selected by the user.
The Factory class should have a method to initialize the block definition.
There should only be one instance of the block definition created, that then gets shared by all the individuals in the population.
A decision was made in the development of the code, to separate the 'structural definition' of the block, from the actual 'genetic material' unique to each individual. The intention was to make the genetic representation of an individual or of a block
The class has several methods to help with the process of initializing a genome, and with mutating or mating; additionally there are 'wrapper' methods for mate, mutate, and evaluate: an 'Individual Definition' calls the this block to call the respective 'Block Definition' for that method.

This BlockDefinition will get instantiated in the user's problem class since it defines the general scope of the evolutionary process.
'''

### packages
from numpy import random as rnd
import numpy as np
import logging

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
    Retains the 6 block characteristics/definitions; the un-instantiated class objects are taken in as input to __init__ method, where they are instanitated to an attribute of the BlockDefinition and where other attributes are created as shortcuts to info from the definition classes.
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
            #self.__dict__[name] = val #TODO maybe change to setattr since that is more commonly used
            setattr(self, name, val)
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


    def get_node_dtype(self, block_material: BlockMaterial, node_index: int, key: str=None):
        '''
        this is a method to quickly grab the data type info at the 'node_index' position of the block_material genome
        
        * if the node_index is for an 'input/output node' then give the data type of the expected input/output
        * otherwise, it's a 'main node'; the 'key' will either be ["inputs", "output", "args"] and will return the respective value in the operator_dict at that node_index
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
        search the genome of the block_material between _min and _max, for a node that outputs the req_dtype.
        return None if we failed to find a matching input.
        
        note _max is exclusive so [_min,_max)
        '''
        if _min is None:
            _min = -1*self.input_count
        if _max is None:
            _max = self.main_count
        
        choices = np.arange(_min, _max)
        for val in exclude:
            choices = np.delete(choices, np.where(choices==val))

        if len(choices) == 0:
            logging.warning("%s - Eliminated all possible input nodes with exclude: %s" % (block_material.id, exclude))
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
            logging.warning("%s - None of the input nodes matched for req_dtype: %s, exclude: %s" % (block_material.id, req_dtype, exclude))
            return None


    def get_random_ftn(self, req_dtype=None, exclude=[], return_all=False):
        '''
        similar to get_random_input but returns a function/primitive that, if given, will output something with the same data type as req_dtype.
        if return_all, it will return all matching functions but in a random order based off a random sample; otherwise it returns just one randomly sampled.
        
        we should only fail to find a matching function, if exclude contains all functions that could match. This assumes that the user has included primitives that output data types that we would want to see in our genome
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
        
        if len(choices) == 0:
            # we have somehow eliminated all possible options
            logging.warning("%s - Eliminated all available operators for req_dtype: %s, and excluding: %s" % (None, req_dtype, exclude))
            return None

        if weights.sum() < 1 - 1e-3:
            # we must have removed some values...normalize
            weights *= 1/weights.sum()

        if return_all:
            return rnd.choice(choices, size=len(choices), replace=False, p=weights)
        else:
            return rnd.choice(choices, p=weights)


    def get_random_arg(self, req_dtype, exclude=[]):
        '''
        similar to get_random_input to find an arg_index that matches the req_dtype
        '''
        choices = []
        for arg_index, arg_type in enumerate(self.arg_types):
            if (arg_type == req_dtype) and (arg_index not in exclude):
                choices.append(arg_index)

        if len(choices) == 0:
            logging.warning("%s - Eliminated all possible arg values for req_dtype: %s, exclude: %s" % (None, req_dtype, exclude))
            return None
        else:
            return rnd.choice(choices)


    def get_actives(self, block_material: BlockMaterial):
        '''
        method will go through and set the attributes block_material.active_nodes and active_args.
        active_nodes will include all output_nodes, a subset of main_nodes and input_nodes.
        '''
        logging.debug("%s - Getting active nodes" % (block_material.id))
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


    def mutate(self, mutant_material: BlockMaterial):
        '''
        wrapper method to call the block's mutate definition
        '''
        logging.debug("%s - Sending to Block Mutate Definition" % (mutant_material.id))
        self.mutate_def.mutate(mutant_material, self)
        self.get_actives(mutant_material)


    def mate(self, parent1: IndividualMaterial, parent2: IndividualMaterial, block_index: int):
        '''
        wrapper method to call the block's mate definition
        '''
        logging.debug("%s+%s - Sending to Block Mate Definition" % (parent1.id, parent2.id))
        children = self.mate_def.mate(parent1, parent2, self, block_index)
        logging.debug("%s+%s - Received %i Children from Block Mate Definition" % (parent1.id, parent2.id, len(children)))
        for child in children:
            self.get_actives(child[block_index])
        return children


    def evaluate(self, block_material: BlockMaterial, training_datapair: ezDataSet, validation_datapair=None):
        '''
        wrapper method to call the block's evaluate definition
        '''
        logging.debug("%s - Sending to Block Evaluate Definition" % (block_material.id))
        # verify that the input data matches the expected datatypes
        for input_dtype, input_data in zip(self.input_dtypes, training_datapair):
            if input_dtype != type(input_data):
                logging.critical("%s - Input data type (%s) doesn't match excted type (%s)" % (block_material.id, type(input_data), input_dtype))
                return None

        self.evaluate_def.reset_evaluation(self, block_material)
        output = self.evaluate_def.evaluate(self, block_material, training_datapair, validation_datapair)
        return output
