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
        # ...uh who added this? i hate this -> TODO
        if rnd.random() < 0.5:
            # do not mutate
            return mutants

        for block_index, block_def in enumerate(indiv_def.block_defs):
            roll = rnd.random()
            if roll < block_def.mutate_def.prob_mutate:
                for _ in range(block_def.num_mutants):
                    mutant_material = deepcopy(indiv_material)
                    block_def.mutate(mutant_material.blocks[block_index])
                    indiv_def.postprocess_evolved_individual(mutant_material, block_index)
                    mutants.append(mutant_material)
        return mutants



class IndividualMutate_RollOnEachBlock_LimitedMutants(IndividualMutate_Abstract):
    '''
    The non-'limited' version will deepcopy the parent on every block so that more
    mutants are produced...instead we will use the same mutant as we 'roll' on each
    block.
    Also it doesn't listen to the block_def.num_mutants, which can be confusing I'm
    sure but no better solution for now.
    '''
    def __init__(self):
        self.prob_mutate = 1.0
        self.num_mutants = 4 # statistically there is a chance we won't get this but at most this number


    def mutate(self,
               indiv_material: IndividualMaterial,
               indiv_def):
        mutants = []
        roll = rnd.random()
        if roll < self.prob_mutate:
            # then let's mutate this individual...
            for _ in range(self.num_mutants):
                # set up new mutants
                mutated = False
                first_mutated_block_index = None
                mutant_material = deepcopy(indiv_material)

                for block_index, block_def in enumerate(indiv_def.block_defs):
                    roll = rnd.random()
                    if roll < block_def.mutate_def.prob_mutate:
                        block_def.mutate(mutant_material.blocks[block_index])
                        indiv_def.postprocess_evolved_individual(mutant_material, block_index)
                        mutated = True
                        if first_mutated_block_index is None:
                            first_mutated_block_index = block_index

                if mutated:
                    indiv_def.postprocess_evolved_individual(mutant_material, first_mutated_block_index)
                    mutants.append(mutant_material)

        return mutants
