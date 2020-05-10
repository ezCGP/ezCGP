'''
root/codes/block_definitions/block_mutate.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
from abc import ABC, abstractmethod
from numpy import random as rnd

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from codes.genetic_material import BlockMaterial
from codes.block_definitions.block_definition import BlockDefinition
from codes.block_definitions.utilities import mutate_methods



class BlockMutate_Abstract(ABC):
    '''
    REQUIREMENTS/EXPECTATIONS
    Block Mutate class:
     * in __init__ will assign a prob_mutate and num_mutants attribute for that block
     * this method will mutate the given individual in-place. do not deepcopy here
     * inputs: instance of IndividualMaterial, integer for the i^th block we want to mutate
     * returns: nothing as the mutation should occur in-place to the given individual
    '''
    def __init__(self):
        pass

    @abstractmethod
    def mutate(self, mutant_material: BlockMaterial, block_def: BlockDefinition):
        pass



class BlockMutate_OptA(BlockMutate_Abstract):
    '''
    TODO
    '''
    def __init__(self):
        self.prob_mutate = 1.0
        self.num_mutants = 4

    def mutate(self, mutant_material: BlockMaterial, block_def: BlockDefinition):
        roll = rnd.random()
        if roll < (1/2):
            mutate_methods.mutate_single_input(mutant_material, block_def)
        else:
            mutate_methods.mutate_single_ftn(mutant_material, block_def)



class BlockMutate_OptB(BlockMutate_Abstract):
    '''
    TODO
    '''
    def __init__(self):
        self.prob_mutate = 1.0
        self.num_mutants = 4


    def mutate(self, mutant_material: BlockMaterial, block_def: BlockDefinition):
        roll = rnd.random()
        if roll < (1/4):
            mutate_methods.mutate_single_input(mutant_material, block_def)
        elif roll < (2/4):
            mutate_methods.mutate_single_argvalue(mutant_material, block_def)
        elif roll < (3/4):
            mutate_methods.mutate_single_argindex(mutant_material, block_def)
        else:
            mutate_methods.mutate_single_ftn(mutant_material, block_def)