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
        logging.debug("%s-%s - Starting Initialize Block" % (None, self.nickname))
        self.meta_def = meta_def()
        for name, val in self.meta_def.__dict__.items():
            # quick way to take all attributes and add to self
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
        logging.debug("%s-%s - Done Initialize Block" % (None, self.nickname))


    def get_lisp(self, block_material: BlockMaterial):
        '''
        the idea is to help with seeding, we have the ability to collapse an individual into a tree
        and then collapse into a lisp string representation

        note we'll be using active nodes a lot here...and don't forget that "active_nodes will include
        all output_nodes, a subset of main_nodes and input_nodes."
        '''
        block_material.lisp = []

        # get actives just in case it has changed since last evaluated...but it really shouldn't have!
        self.get_actives(block_material)

        # each output will have it's own tree
        #for ith_output in range(self.output_count):

        # first going to create a dictionary of the active nodes inputs
        _active_dict = {}
        #output_node = self.main_count+ith_output
        #_active_dict['output'] = block_material[output_node]
        for ith_node in reversed(self.active_nodes):
            if (ith_node<0) or (ith_node>=self.main_count):
                #input or ouptput node
                continue
            func = block_material[ith_node]['function']
            inputs = block_material[ith_node]['inputs']
            args = block_material[ith_node]['args']

            # now start to shape it into a lisp
            lisp = ['%s' % func.__name__]
            for _input in inputs:
                # attach an 'n' to remind us that this is a node number and not an arg
                # later we'll go through and replace each node with it's own entry in _active_dict
                #lisp.append('%in' % _input)
                # TODO: consider just leaving it as '-1n' or something...converting to data type and
                # passing as string and then removing quotes n spaces will likely make it unusable to compare anyways
                pass
            for _arg in args:
                lisp.append('%s' % str(block_material.args[_arg]))

            # and now throw the lisp into our _active_dict (tree)
            _active_dict['%i' % ith_node] = lisp

        # at this point we have the ith node and arg values for each active node
        # now we'll go through the list and replace each node with each entry from the dict
        # this is how we slowly build out the branches of the trees
        for ith_node in self.active_args:
            lisp = _active_dict[str(ith_node)]
            new_lisp = []
            for i, val in enumerate(lisp):
                if i == 0:
                    # 0th position in lisp should be the function. keep and append.
                    pass
                elif val.endswith('n'):
                    if int(val[:-1]) < 0:
                        # input node so we want to instead pass in the datatype we expect
                        # -1th genome is 0th input
                        # -2nd genome is 1th input... genome_node*-1 - 1 = input_node
                        val = self.input_dtypes[(i*-1)-1]
                    else:
                        # then it's a node number, replace with that node's new_lisp
                        val = _active_dict[str(val[:-1])] #[:-1] to remove 'n'
                else:
                    # then it's an arg and we pass it as such
                    pass
                new_lisp.append(val)
            # replace lisp with new_lisp in the dict
            _active_dict[str(ith_node)] = new_lisp

        # now all that should be left are the output nodes
        # each output node produces it's own tree so it's own final lisp
        for ith_output in range(self.output_count):
            final_node = block_material[self.main_count+ith_output]
            # convert the final list into a string
            lisp_str = str(_active_dict[final_node])
            # since it was a list of strings, there will be a mess of quotes inside
            # so replace any quotes, and spaces while we're at it
            lisp_str = lisp_str.replace("'","").replace('"','').replace(" ", "")
            # that's our tree. so append to lisp
            block_material.lisp.append(lisp_str)


    def get_node_dtype(self, block_material: BlockMaterial, node_index: int, key: str=None):
        '''
        this is a method to quickly grab the data type info at the 'node_index' position of the block_material genome
        
        * if the node_index is for an 'input/output node' then give the data type of the expected input/output
        * otherwise, it's a 'main node'; the 'key' will either be ["inputs", "output", "args"] and will return the respective value in the operator_dict at that node_index
        '''
        logging.debug("%s - Inside get_node_dtype; node_index: %i, key: %s" % (block_material.id, node_index, key))
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
        logging.debug("%s - Inside get_random_input; req_dtype: %s, _min: %s, _max: %s, exclude: %s" % (block_material.id, req_dtype, _min, _max, exclude))
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
                logging.debug("%s - trying to match index %i with %s to %s" % (block_material.id, input_index, input_dtype, req_dtype))
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
        logging.debug("%s-%s - Inside get_random_ftn; req_dtype: %s, exclude: %s, return_all: %s" % (None, self.nickname, req_dtype, exclude, return_all))
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
            logging.warning("%s-%s - Eliminated all available operators for req_dtype: %s, and excluding: %s" % (None, self.nickname, req_dtype, exclude))
            return None

        if weights.sum() < 1 - 1e-3: #arbitrarily chose 1e-3 to account for rounding errors
            # we must have removed some values...normalize
            weights /= weights.sum()

        if return_all:
            return rnd.choice(choices, size=len(choices), replace=False, p=weights)
        else:
            return rnd.choice(choices, p=weights)


    def get_random_arg(self, req_dtype, exclude=[]):
        '''
        similar to get_random_input to find an arg_index that matches the req_dtype
        '''
        logging.debug("%s-%s - Inside get_random_arg; req_dtype: %s, exclude: %s" % (None, self.nickname, req_dtype, exclude))
        choices = []
        for arg_index, arg_type in enumerate(self.arg_types):
            if (arg_type == req_dtype) and (arg_index not in exclude):
                choices.append(arg_index)

        if len(choices) == 0:
            logging.warning("%s-%s - Eliminated all possible arg values for req_dtype: %s, exclude: %s" % (None, self.nickname, req_dtype, exclude))
            return None
        else:
            return rnd.choice(choices)


    def get_actives(self, block_material: BlockMaterial):
        '''
        method will go through and set the attributes block_material.active_nodes and active_args.
        active_nodes will include all output_nodes, a subset of main_nodes and input_nodes.
        '''
        logging.info("%s - Inside get_actives" % (block_material.id))
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
            else:
                pass
            
        # sort
        block_material.active_nodes = sorted(list(block_material.active_nodes))
        logging.debug("%s - active nodes: %s" % (block_material.id, block_material.active_nodes))
        block_material.active_args = sorted(list(block_material.active_args))
        logging.debug("%s - active args: %s" % (block_material.id, block_material.active_args))


    def mutate(self, mutant_material: BlockMaterial):
        '''
        wrapper method to call the block's mutate definition
        '''
        logging.info("%s - Sending to Block Mutate Definition" % (mutant_material.id))
        self.mutate_def.mutate(mutant_material, self)


    def mate(self, parent1: IndividualMaterial, parent2: IndividualMaterial, block_index: int):
        '''
        wrapper method to call the block's mate definition
        '''
        logging.info("%s+%s-%s - Sending to Block Mate Definition" % (parent1.id, parent2.id, self.nickname))
        children = self.mate_def.mate(parent1, parent2, self, block_index)
        logging.debug("%s+%s-%s - Received %i Children from Block Mate Definition" % (parent1.id, parent2.id, self.nickname, len(children)))
        return children


    def evaluate(self, block_material: BlockMaterial, training_datapair: ezDataSet, validation_datapair=None):
        '''
        wrapper method to call the block's evaluate definition
        NOTE: we take the output and attach to block_material in postprocess_evaluated_block BUT ALSO return the output to the IndividualEvaluate method
        '''
        logging.debug("%s - Sending to Block Evaluate Definition" % (block_material.id))
        # verify that the input data matches the expected datatypes
        for input_dtype, input_data in zip(self.input_dtypes, training_datapair):
            if input_dtype != type(input_data):
                logging.critical("%s - Input data type (%s) doesn't match excted type (%s)" % (block_material.id, type(input_data), input_dtype))
                return None

        self.evaluate_def.reset_evaluation(block_material)
        logging.debug("%s - Before evaluating list active nodes: %s, and args %s" % (block_material.id, block_material.active_nodes, block_material.active_args))
        output = self.evaluate_def.evaluate(block_material, self, training_datapair, validation_datapair)
        self.evaluate_def.postprocess_evaluated_block(block_material, output)
        return output
