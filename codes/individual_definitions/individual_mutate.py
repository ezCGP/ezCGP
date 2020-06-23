'''
root/code/individual_definitions/individual_mutate.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
from abc import ABC, abstractmethod
from numpy import random as rnd
from copy import deepcopy

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from codes.genetic_material import IndividualMaterial
#from codes.individual_definitions.individual_definition import IndividualDefinition #circular dependecy
from codes.utilities.custom_logging import ezLogging



class IndividualMutate_Abstract(ABC):
    '''
    REQUIREMENTS/EXPECTATIONS
    Individual Mutate class:
     * deepcopies should always happen at this level and the copied individual sent to the blocks to be mutated in-place
     * RE setting need_evaluate to True after mutation, this is expected to occur at the mutate_method level because it is possible for some mutations
        to not mutate active nodes so need_evaluate could remain False
     * inputs: instance of IndividualDefinition and instance of IndividualMaterial
     * returns: a list of new mutated individuals or an empty list
    '''
    def __init__(self):
        pass

    @abstractmethod
    def mutate(self,
               indiv_material: IndividualMaterial,
               indiv_def): #: IndividualDefinition):
        pass


    def set_need_evaluate(self,
                          mutant_material: IndividualMaterial,
                          mutated_block_index: int):
        # going to assume we always mutate an active part of the genome so every block needs to be re-evaluated
        # TODO doc more
        for block_index, block_material in enumerate(mutant_material.blocks):
            if block_index >= mutated_block_index:
                block_material.need_evaluate = True
    
    
    def post_process(self,
                     mutant_material: IndividualMaterial,
                     indiv_def, #: IndividualDefinition,
                     mutated_block_index: int):
        # set need_evaluate
        # and set ne ids?
        mutant_material.set_id()
        for block_index, block_material in enumerate(mutant_material.blocks):
            #block_material.set_id(mutant_material.id) #moved to individual_mutate.set_id()
            if block_index >= mutated_block_index:
                block_material.need_evaluate = True
        indiv_def.blocks[mutated_block_index].get_actives(block_material)



class IndividualMutate_RollOnEachBlock(IndividualMutate_Abstract):
    '''
    TODO
    '''
    def __init__(self):
        pass

    def mutate(self,
               indiv_material: IndividualMaterial,
               indiv_def): #: IndividualDefinition):
        mutants = []
        for block_index, block_def in enumerate(indiv_def.block_defs):
            roll = rnd.random()
            if roll < block_def.mutate_def.prob_mutate:
                for _ in range(block_def.num_mutants):
                    mutant_material = deepcopy(indiv_material)
                    block_def.mutate(mutant_material.blocks[block_index])
                    #self.set_need_evaluate(mutant_material, block_index)
                    indiv_def.postprocess_evolved_individual(mutant_material, block_index)
                    mutants.append(mutant_material)
        return mutants