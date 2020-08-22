'''
root/code/factory.py

Overview:
factory class will be tasked with building/__init__ all the other classes.
that way we can wrap other debugging + logging items around the init of each class

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
import os
import re
import numpy as np
from copy import deepcopy
from typing import List

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from codes.population import PopulationDefinition
from codes.genetic_material import IndividualMaterial, BlockMaterial
from codes.individual_definitions.individual_definition import IndividualDefinition
from codes.block_definitions.block_definition import BlockDefinition
from codes.utilities.custom_logging import ezLogging



class FactoryDefinition():
    '''
    TODO
    '''
    def __init__(self):
        pass


    def build_population(self,
                         indiv_def: IndividualDefinition,
                         population_size: int,
                         genome_seeds: List=[[]],
                         node_rank: int=0,
                        ):
        '''
        TODO
        '''
        my_population = PopulationDefinition(population_size)

        for i, genome_seed in genome_seeds:
            '''# should be a filepath
            if genome_seed.endswith("pkl"):
                with open(genome_seed, "rb") as f:
                    indiv = pkl.load(f)
                if isinstance(indiv, IndividualMaterial):
                    indiv.set_id("seededIndiv%i" % i)
                elif "lisp" in genome_seed:
                    # TODO which block?
                    indiv = build_block_from_lisp(block_def, lisp=indiv, indiv_id="seededIndiv%i-%i" % (node_rank,i))
                else:
                    ezLogging.error("unable to interpret genome seed")
                    return None'''
            indiv = build_individual_from_seed(indiv_def,
                                               genome_seed,
                                               indiv_id="seededIndiv%i-%i" % (node_rank,i))
            if isinstance(indiv, IndividualMaterial):
                # if build_individual failed then we don't want to add to population
                my_population.population.append(indiv)

        for i in range(len(my_population.population), population_size):
            indiv = self.build_individual(indiv_def,
                                          indiv_id="initPop%i-%i" % (node_rank,i))
            my_population.population.append(indiv)

        return my_population


    '''
    def build_subpopulation(self, indiv_def: IndividualDefinition, population_size, number_subpopulation=None, subpopulation_size=None):
        ''
        TODO
        ''
        my_population = SubPopulationDefinition(population_size, number_subpopulation, subpopulation_size)
        for ith_subpop, subpop_size in my_population.subpop_size:
            for _ in range(subpop_size):
                indiv = self.build_individual(indiv_def)
                my_population[ith_subpop].append(indiv)
        return my_population
    '''


    def build_individual(self, indiv_def: IndividualDefinition, indiv_id=None):
        '''
        TODO
        '''
        indiv_material = IndividualMaterial()
        indiv_material.set_id(indiv_id)
        for block_def in indiv_def.block_defs:
            block_material = self.build_block(block_def, indiv_id=indiv_id)
            indiv_material.blocks.append(block_material)
        return indiv_material


    def build_individual_from_seed(self, indiv_def: IndividualDefinition, block_seeds: List[str], indiv_id=None):
        '''
        block_seeds will be a list of file paths. number of blocks in individual need to match block_seed length
        If we don't have a seed for a block, set it to None like [None, None, seed.npz]
        '''
        if (isinstance(block_seeds, str)) and (os.path.exists(block_seeds)):
            # then it's likely a pickled IndividualMaterial instance from a pervious run or text file of lisps
            # load the whole thing as an individual but then check to make sure all the indiv/block defs match
            try:
                # adding a try except loop because any number of things could go wrong here so easier to 
                # try and catch anything...also makes it easy to catch when seeded material doesn't match defs
                if block_seeds.endswith(".pkl"):
                    with open(block_seeds, "rb") as f:
                        indiv_material = pkl.load(f)
                    if not isinstance(indiv_material, IndividualMaterial):
                        raise Exception("pickled file was not an IndividualMaterial type but %s" % (type(indiv_material)))
                    indiv_material.set_id(indiv_id)
                    # now check each block to make sure it matches respective definition
                    for ith_block in range(indiv_def.block_count):
                        if not self.validate_material_wDefinition(indiv_def[ith_block], indiv_material[ith_block]):
                            raise Exception("%ith block material does not match block definition")
                elif ("lisp" in block_seeds) and (block_seeds.endswith(".txt")):
                    # TODO read several lisps from text file
                    block_seeds = []
                    if len(block_seeds) != indiv_def.block_count:
                        raise Exception("number of lisps in seed file doesn't match IndividualDefinition")
                    else:
                        pass
                else:
                    raise Exception("block_seeds wasnt pkl or txt file")
            except Exception as err:
                ezLogging.error("%s - for seed %s" % (err, block_seeds))
                indiv_material = None
                
        if isinstance(block_seeds, list):
            indiv_material = IndividualMaterial()
            indiv_material.set_id(indiv_id)
            for ith_block in range(indiv_def.block_count):
                # trying to refrain from zipping block_defs and block_seeds because it won't error if len don't match
                block_def = indiv_def[ith_block]
                block_seed = block_seeds[ith_block]
                if block_seed is None:
                    block_material = self.build_block(block_def, indiv_id=indiv_id)
                else:
                    try:
                        if os.path.exists(block_seed):
                            if block_seed.endswith(".pkl"):
                                with open(genome_seed, "rb") as f:
                                    block_material = pkl.load(f)
                                if not isinstance(block_material, BlockMaterial):
                                    raise Exception("pickled file was not an BlockMaterial type but %s" % (type(block_material)))
                            elif ("lisp" in block_seed) and (block_seed.endswith(".txt")):
                                # TODO how to read lips from text file
                                block_seed = ""
                            else:
                                raise Exception("block_seed wasnt pkl or txt file")
                        if isinstance(block_seed, str):
                            block_material = self.build_block_from_lisp(block_def, block_seed, indiv_id)
                        if not self.validate_material_wDefinition(block_def, block_material):
                            raise Exception("%ith block material does not match block definition")
                    except Exception as err:
                        ezLogging.error("%s - for %th block and seed %s" % (err, ith_block, block_seed))
                        indiv_material = None
                        break
                indiv_material.blocks.append(block_material)
                
        return indiv_material


    def build_block(self, block_def: BlockDefinition, indiv_id):
        '''
        TODO
        '''
        block_material = BlockMaterial(block_def.nickname)
        block_material.set_id(indiv_id)
        self.fill_args(block_def, block_material)
        self.fill_genome(block_def, block_material)
        block_def.get_actives(block_material)
        return block_material


    def build_block_from_lisp(self, block_def: BlockDefinition, lisp: str, indiv_id):
        '''
        the expectation here is that lisp is the string tree representation (not a file holding the str)
        that follows the format of how we build out a lisp in codes.block_definitions.block_definition.get_lisp()

        shoud look something like: [func1,[func2,-2n,-1n],-1n]
        
        we also can handle cases where we are 'reusing' the output of a node...thanks to this line
            ```lisp = lisp.replace(_active_dict[ith_node], "%in" % ith_node)```
        try with: lisp = '[mult,-1n,[mult,[sub,[mult,-1n,-1n],-2n],[sub,[mult,-1n,-1n],-2n]]]'

        NOTE: I think this currently only works if the block has 1 output!
        '''
        _active_dict = {}
        ith_node = 0
        while True:
            # from the start of the string, keep looking for lists []
            match = re.search("\[[0-9A-Za-z_\-\s.,']+\]", lisp)
            if match is None:
                # no more lists inside lisp. so we're done
                break
                # get the single element lisp
                _active_dict[ith_node] = lisp[match.start(): match.end()]
                # now replace that element with the node number
                # add 'n' to distinguish from arg value
                lisp = lisp.replace(_active_dict[ith_node], "%in" % ith_node)
                # increment to next node
                ith_node +=1

                if ith_node >= 10**3:
                    # very unlikely to have more than 1000 nodes...prob something went wrong
                    ezLogging.error("something went wrong")
                    break

        # now build the individual
        block_material = BlockMaterialz(block_def.nickname)
        block_material.set_id(indiv_id)
        block_material.args = [None]*block_def.arg_count
        block_material.genome = [None]*block_def.genome_count
        block_material.genome[(-1*block_def.input_count):] = ["InputPlaceholder"]*block_def.input_count

        ith_active_node = -1
        active_main_nodes = sorted(np.random.choice(range(block_def.main_count), size=len(_active_dict), replace=False))
        for node_index in range(block_def.main_count):
            if node_index in active_main_nodes:
                ith_active_node +=1
                # fill node with what we got from the lisp
                lisp = _active_dict[ith_active_node].strip('][').split(',')
                for ftn in block_def.operator_dict.keys():
                    if ftn.__name__ == lisp[0]:
                        # we matched our lisp ftn with entry in operatordict
                        input_index = []
                        arg_index = []
                        ith_input = -1
                        ith_arg = -1
                        args_used = []
                        # now grab what we have in the lisp and make sure they match
                        for val in lisp[1:]: # [1:] #ignores the ftn in 0th element
                            if val.endswith('n'):
                                # then it's a node index
                                ith_input +=1
                                extracted_val = int(val[:-1]) # [:-1] to remove 'n'
                                if extracted_val >= 0:
                                    # then it's a main node
                                    input_index.append(active_main_nodes[int(extracted_val)])
                                    # verify that the data types match
                                    incoming_dtype = block_def.get_node_dtype(self,
                                                                              block_material,
                                                                              node_index=input_index[ith_input],
                                                                              key='output')
                                    expected_dtype = block_def.operator_dict[ftn]["inputs"][ith_input]
                                    if incoming_dtype != expected_dtype:
                                        ezLogging.error("error in genome seeding...mismatching incoming + given data types")
                                        import pdb; pdb.set_trace()
                                        return None
                                    else:
                                        # all good
                                        pass
                                else:
                                    # then it's an input node
                                    input_index.append(extracted_val)
                            else:
                                # then it's an arg value
                                ith_arg +=1
                                req_arg_type = block_def.operator_dict[ftn]["args"][ith_arg]
                                poss_arg = block_def.get_random_arg(req_arg_type, exclude=args_used)
                                if poss_arg is None:
                                    ezLogging.error("can't find matching arg type in seeding")
                                    import pdb; pdb.set_trace()
                                    return None
                                arg_index.append(poss_arg)
                                args_used.append(poss_arg)
                                block_material.args[poss_arg] = req_arg_type(value=val)

                        block_material[node_index] =  {"ftn": ftn,
                                                       "inputs": input_index,
                                                       "args": arg_index}
                    else:
                        # ftn doesn't match our lisp
                        continue
            else:
                # then this is not an active main node...just fill it normally then
                # copy paste from fill_nodes
                ftns = block_def.get_random_ftn(return_all=True)
                for ftn in ftns:
                    # find inputs
                    input_dtypes = block_def.operator_dict[ftn]["inputs"]
                    input_index = [None]*len(input_dtypes)
                    for ith_input, input_dtype in enumerate(input_dtypes):
                        input_index[ith_input] = block_def.get_random_input(block_material, req_dtype=input_dtype, _max=node_index)
                    if None in input_index:
                        # failed to fill it in; try another ftn
                        continue
                    else:
                        pass
                    # find args
                    arg_dtypes = block_def.operator_dict[ftn]["args"]
                    arg_index = [None]*len(arg_dtypes)
                    for ith_arg, arg_dtype in enumerate(arg_dtypes):
                        arg_index[ith_arg] = block_def.get_random_arg(req_dtype=arg_dtype)
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
                    import pdb; pdb.set_trace()
                    return None

        # output node
        # currently only works for 1 output node
        block_material[main_count] = active_main_nodes[-1]

        # now finish filling in the args
        for arg_index, arg_type in enumerate(block_def.arg_types):
            if block_material.args[arg_index] is None:
                block_material.args[arg_index] = arg_type()
            else:
                continue

        block_def.get_actives(block_material)
        return block_material
    
    
    def validate_material_wDefinition(self, block_def: BlockDefinition, block_material: BlockMaterial):
        '''
        assuming we got a seeded individual/block passed into block_material,
        it would be wise to try and make sure that the material fits into the
        BlockDefinition for this simulation/run
         * ShapeMeta matches: genome_size, arg_size, input/output datatypes
         * Operator matches: primitives in operator dictionary
         * Argument matches: datatypes in genome match what is available
        
        return True or False
        '''
        pass # TODO
        return True


    def fill_args(self, block_def: BlockDefinition, block_material: BlockMaterial):
        '''
        TODO
        '''
        block_material.args = [None]*block_def.arg_count
        for arg_index, arg_type in enumerate(block_def.arg_types):
            block_material.args[arg_index] = arg_type()


    def fill_genome(self, block_def: BlockDefinition, block_material: BlockMaterial):
        '''
        TODO
        '''
        block_material.genome = [None]*block_def.genome_count
        block_material.genome[(-1*block_def.input_count):] = ["InputPlaceholder"]*block_def.input_count

        # fill main nodes
        for node_index in range(block_def.main_count):
            ftns = block_def.get_random_ftn(return_all=True)
            for ftn in ftns:
                # find inputs
                input_dtypes = block_def.operator_dict[ftn]["inputs"]
                input_index = [None]*len(input_dtypes)
                for ith_input, input_dtype in enumerate(input_dtypes):
                    input_index[ith_input] = block_def.get_random_input(block_material, req_dtype=input_dtype, _max=node_index)
                if None in input_index:
                    # failed to fill it in; try another ftn
                    continue
                else:
                    pass

                # find args
                arg_dtypes = block_def.operator_dict[ftn]["args"]
                arg_index = [None]*len(arg_dtypes)
                for ith_arg, arg_dtype in enumerate(arg_dtypes):
                    arg_index[ith_arg] = block_def.get_random_arg(req_dtype=arg_dtype)
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
        for ith_output, node_index in enumerate(range(block_def.main_count, block_def.main_count+block_def.output_count)):
            req_dtype = block_def.output_dtypes[ith_output]
            block_material[node_index] = block_def.get_random_input(block_material, req_dtype=req_dtype, _min=0, _max=block_def.main_count)