'''
root/codes/block_definitions/utilities/mutate_methods.py

Overview:
One of the important coding principles here is to try and preserve life as much as possible. Try to avoid doing any mutation that we know will not work, like connecting two nodes whose data types do not match...that does make coding this all a bit more complicated but it should help with the efficiency of the program as well...the beauty of strongly typed!

Note that these BlockMaterials have already been deepcopied from the original individual so we shouldn't be hesitant about editing the genome in place.
'''

### packages
from numpy import random as rnd
import numpy as np
import logging

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.genetic_material import BlockMaterial
from codes.block_definitions.block_definition import BlockDefinition


def mutate_single_input(mutant_material: BlockMaterial, block_def: BlockDefinition):
    '''
    pick nodes at random and mutate the input-index until an active node is selected
    when mutating inputs, it will look for a node that outputs the matching datatype of the current node's input
    so it can fail at doing so and won't mutate that node
    '''
    logging.info("%s - Inside mutate_single_input" % (mutant_material.id))
    choices = np.arange(block_def.main_count+block_def.output_count)
    choices = rnd.choice(choices, size=len(choices), replace=False) #randomly reorder
    for node_index in choices:
        if node_index < block_def.main_count:
            # then we are mutating a main-node (expect a node-dict)
            num_inputs_into_node = len(mutant_material[node_index]["inputs"])
            ith_input = rnd.choice(np.arange(num_inputs_into_node))
            current_input_index = mutant_material[node_index]["inputs"][ith_input]
            req_dtype = block_def.get_node_dtype(mutant_material, node_index, "inputs")[ith_input]
            new_input = block_def.get_random_input(mutant_material, req_dtype=req_dtype, max_=node_index, exclude=[current_input_index])
            if new_input is None:
                # failed to find new input, will have to try another node to mutate
                continue
            else:
                mutant_material[node_index]["inputs"][ith_input] = new_input
                logging.debug("%s - Mutated node %i; ori_input: %i, new_input: %i" % (mutant_material.id, node_index, current_input_index, new_input))
                if node_index in mutant_material.active_nodes:
                    # active_node finally mutated
                    logging.debug("%s - Mutated node %i - active" % (mutant_material.id, node_index))
                    mutant_material.need_evaluate = True
                    break
                else:
                    logging.debug("%s - Mutated node %i - inactive" % (mutant_material.id, node_index))
                    pass
        else:
            # then we are mtuating an output-node (expect a int index value)
            current_output_index = mutant_material[node_index]
            req_dtype = block_def.output_dtypes[node_index-block_def.main_count]
            new_output_index = block_def.get_random_input(mutant_material, req_dtype=req_dtype, min_=0, exclude=[current_output_index])
            if new_output_index is None:
                # failed to find new node
                continue
            else:
                mutant_material[node_index] = new_output_index
                logging.debug("%s - Mutated node %i; ori_input: %i, new_input: %i" % (mutant_material.id, node_index, current_output_index, new_output_index))
                # active_node finally mutated
                logging.debug("%s - Mutated node %i - active" % (mutant_material.id, node_index))
                mutant_material.need_evaluate = True
                break


def mutate_single_ftn(mutant_material: BlockMaterial, block_def: BlockDefinition):
    '''
    pick nodes at random and mutate the ftn-index until an active node is selected
    will mutate the function to anything with matching input/arg dtype.
    if the expected input datatypes don't match the current genome,
    it will find a new input/arg that will match
    '''
    logging.info("%s - Inside mutate_single_ftn" % (mutant_material.id))
    choices = np.arange(block_def.main_count)
    choices = rnd.choice(choices, size=len(choices), replace=False) #randomly reorder
    for node_index in choices:
        # note, always will be a main_node
        current_ftn = mutant_material[node_index]["ftn"]
        req_output_dtype = block_def.operator_dict[current_ftn]["output"]
        new_ftn = block_def.get_random_ftn(req_dtype=req_output_dtype, exclude=[current_ftn])
        logging.debug("%s - Mutated node %i - possible new ftn: %s" % (mutant_material.id, node_index, new_ftn))

        # make sure input_dtypes match
        req_input_dtypes = block_def.operator_dict[new_ftn]["inputs"]
        new_inputs = [None]*len(req_input_dtypes)
        for input_index in mutant_material[node_index]["inputs"]:
            exist_input_dtype = block_def.get_node_dtype(mutant_material, input_index, "output") #instead of verify from current node, goes to input
            for ith_input, (new_input, input_dtype) in enumerate(zip(new_inputs, req_input_dtypes)):
                if (new_input is None) and (input_dtype == exist_input_dtype):
                    # then we can match an existing input with one of our required inputs
                    new_inputs[ith_input] = input_index
                else:
                    pass
        # now try and fill in anything still None
        for ith_input, (new_input, input_dtype) in enumerate(zip(new_inputs, req_input_dtypes)):
            if new_input is None:
                new_inputs[ith_input] = block_def.get_random_input(mutant_material, req_dtype=input_dtype, max_=node_index)
        # if there is still 'None' then we failed to fit this ftn in...try another ftn
        if None in new_inputs:
            continue
        else:
            logging.debug("%s - Mutated node %i - possible new inputs: %s" % (mutant_material.id, node_index, new_inputs))

        # make sure arg_dtypes match
        req_arg_dtypes = block_def.operator_dict[new_ftn]["args"]
        new_args = [None]*len(req_arg_dtypes)
        exist_arg_dtypes = block_def.get_node_dtype(mutant_material, node_index, "args")
        for arg_index, exist_arg_dtype in zip(mutant_material[node_index]["args"], exist_arg_dtypes):
            for ith_arg, (new_arg, req_arg_dtype) in enumerate(zip(new_args, req_arg_dtypes)):
                if (new_arg is None) and (req_arg_dtypes == exist_arg_dtype):
                    new_args[ith_arg] = arg_index
                else:
                    pass
        # now try and fill in anything still None
        for ith_arg, (new_arg, req_arg_dtype) in enumerate(zip(new_args, req_arg_dtypes)):
            if new_arg is None:
                new_args[ith_arg] = block_def.get_random_arg(req_dtype=req_arg_dtype)
        # if there is still 'None' then we failed to fit this ftn ...try another ftn
        if None in new_args:
            continue
        else:
            logging.debug("%s - Mutated node %i - possible new args: %s" % (mutant_material.id, node_index, new_args))

        # at this point we found a ftn and fit inputs and args
        mutant_material[node_index]["ftn"] = new_ftn
        mutant_material[node_index]["inputs"] = new_inputs
        mutant_material[node_index]["args"] = new_args
        logging.debug("%s - Mutated node %i; old_ftn: %s, new_ftn: %s, new_inputs: %s, new_args %s" % (mutant_material.id, node_index, current_ftn, new_ftn, new_inputs, new_args))
        if node_index in mutant_material.active_nodes:
            # active_node finally mutated
            logging.debug("%s - Mutated node %i - active" % (mutant_material.id, node_index))
            mutant_material.need_evaluate = True
            break
        else:
            logging.debug("%s - Mutated node %i - inactive" % (mutant_material.id, node_index))
            pass


def mutate_single_argindex(mutant_material: BlockMaterial, block_def: BlockDefinition):
    '''
    search through the args and try to find a matching arg_type and use that arg index instead
    '''
    logging.info("%s - Inside mutate_single_argindex" % (mutant_material.id))
    if len(mutant_material.active_args) > 0:
        # then there is something to mutate
        choices = [] # need to find those nodes with 'args' filled
        #weights = [] # option to sample node_index by the number of args for each node
        for node_index in range(block_def.main_count):
            if len(mutant_material[node_index]["args"]) > 0:
                choices.append(node_index)
                #weights.append(len(mutant_material[node_index]["args"]))
            else:
                pass

        choices = rnd.choice(choices, size=len(choices), replace=False) #randomly reorder
        for node_index in choices:
            ith_arg = rnd.choice(np.arange(len(mutant_material[node_index]["args"])))
            current_arg = mutant_material[node_index]["args"][ith_arg]
            arg_dtype = block_def.get_node_dtype(mutant_material, node_index, "args")[ith_arg]
            new_arg = block_def.get_random_arg(arg_dtype, exclude=[current_arg])
            if new_arg is None:
                # failed to find a new_arg
                continue
            else:
                mutant_material[node_index]["args"][ith_arg] = new_arg
                logging.info("%s - Mutated node %i; ori arg index: %i, new arg index: %i" % (mutant_material.id, node_index, current_arg, new_arg))
                if node_index in mutant_material.active_nodes:
                    # active_node finally mutated
                    logging.debug("%s - Mutated node %i - active" % (mutant_material.id, node_index))
                    mutant_material.need_evaluate = True
                    break
                else:
                    logging.debug("%s - Mutated node %i - inactive" % (mutant_material.id, node_index))
    else:
        # won't actually mutate
        logging.warning("%s - No active args to mutate" % (mutant_material.id))


def mutate_single_argvalue(mutant_material: BlockMaterial, block_def: BlockDefinition):
    '''
    instead of looking for a different arg index in .args with the same arg type,
    mutate the value stored in this arg index.
    '''
    logging.info("%s - Inside mutate_single_argvalue" % (mutant_material.id))
    if len(mutant_material.active_args) > 0:
        # if block has arguments, then there is something to mutate
        choices = np.arange(self.args_count)
        choices = rnd.choice(choices, size=len(choices), replace=False) #randomly reorder
        for arg_index in choices:
            mutant_material.args[arg_index].mutate()
            logging.info("%s - Mutated node %i; new arg value: %s" % (mutant_material.id, node_index, mutant_material.args[arg_index]))
            if arg_index in mutant_material.active_args:
                # active_arg finally mutated
                logging.debug("%s - Mutated node %i - active" % (mutant_material.id, node_index))
                mutant_material.need_evaluate = True
                break
            else:
                logging.debug("%s - Mutated node %i - inactive" % (mutant_material.id, node_index))
    else:
        # won't actually mutate
        logging.warning("%s - No active args to mutate" % (mutant_material.id))