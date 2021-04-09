'''
root/codes/block_definitions/mutate/block_mutate.py

Overview:
Pretty simple class; basically just a wrapper to the mutate_methods. Deepcopies of the original individual happened at the IndividualMutate_Abstract level so we don't have to do anything fancy here...just pass the mutant along.

Rules:
Will need the probability that a block will mutate, and then how many mutants it should create.
'''

### packages
from abc import ABC, abstractmethod
from numpy import random as rnd

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.genetic_material import BlockMaterial
#from codes.block_definitions.block_definition import BlockDefinition #circular dependecy
from codes.block_definitions.utilities import mutate_methods
from codes.utilities.custom_logging import ezLogging



class BlockMutate_Abstract(ABC):
    '''
    the user doesn't have to make their own __init__ method if they're happy with the attributes in this init
    
    if they do make their own __init__, they shoudln't need to call BlockMutate_Abstract.__init__(self)
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockMutate_Abstract Class" % (None, None))
        self.prob_mutate = 1.0
        self.num_mutants = 4

    @abstractmethod
    def mutate(self, mutant_material: BlockMaterial, block_def): #: BlockDefinition):
        pass



class BlockMutate_OptA(BlockMutate_Abstract):
    '''
    Good for things like symbolic regression with NO args since we only mutate the input node connections or the primitive function used.
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockMutate_OptA Class" % (None, None))
        self.prob_mutate = 1.0
        self.num_mutants = 4

    def mutate(self, mutant_material: BlockMaterial, block_def): #: BlockDefinition):
        roll = rnd.random()
        ezLogging.info("%s - Sending block to mutate; roll: %f" % (mutant_material.id, roll))
        if roll < (1/2):
            mutate_methods.mutate_single_input(mutant_material, block_def)
        else:
            mutate_methods.mutate_single_ftn(mutant_material, block_def)



class BlockMutate_OptB(BlockMutate_Abstract):
    '''
    Good to be used for something like symbolic regression WITH args since this inclues mutate methods for args.
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockMutate_OptB Class" % (None, None))
        self.prob_mutate = 1.0
        self.num_mutants = 4


    def mutate(self, mutant_material: BlockMaterial, block_def): #: BlockDefinition):
        roll = rnd.random()
        ezLogging.info("%s - Sending block to mutate; roll: %f" % (mutant_material.id, roll))
        if roll < (1/4):
            mutate_methods.mutate_single_input(mutant_material, block_def)
        elif roll < (2/4):
            mutate_methods.mutate_single_argvalue(mutant_material, block_def)
        elif roll < (3/4):
            mutate_methods.mutate_single_argindex(mutant_material, block_def)
        else:
            mutate_methods.mutate_single_ftn(mutant_material, block_def)


class BlockMutate_SimGAN(BlockMutate_Abstract):
    '''
    Good to be used for something like symbolic regression WITH args since this inclues mutate methods for args.
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockMutate_OptB Class" % (None, None))
        self.prob_mutate = 1.0 # TODO: change to 0.5
        self.num_mutants = 4


    def mutate(self, mutant_material: BlockMaterial, block_def): #: BlockDefinition):
        roll = rnd.random()
        ezLogging.info("%s - Sending block to mutate; roll: %f" % (mutant_material.id, roll))
        if roll < (1/4):
            mutate_methods.mutate_single_input(mutant_material, block_def)
        elif roll < (2/4):
            mutate_methods.mutate_single_argvalue(mutant_material, block_def)
        elif roll < (3/4):
            mutate_methods.mutate_single_argindex(mutant_material, block_def)
        else:
            mutate_methods.mutate_single_ftn(mutant_material, block_def, dont_exclude=True)


class BlockMutate_OptB_4Blocks(BlockMutate_OptB):
    '''
    just wanted to decrease prob of mutate by 1/number of blocks
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockMutate_OptB_4Blocks Class" % (None, None))
        self.prob_mutate = 0.33 #0.25
        self.num_mutants = 2 #4



class BlockMutate_NoFtn(BlockMutate_Abstract):
    '''
    used for guassian sum...we only have 1 primitive so no point in mutating ftns
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockMutate_NoFtn Class" % (None, None))
        self.prob_mutate = 1.0
        self.num_mutants = 4


    def mutate(self, mutant_material: BlockMaterial, block_def): #: BlockDefinition):
        roll = rnd.random()
        ezLogging.info("%s - Sending block to mutate; roll: %f" % (mutant_material.id, roll))
        if roll < (1/3):
            mutate_methods.mutate_single_input(mutant_material, block_def)
        elif roll < (2/3):
            mutate_methods.mutate_single_argvalue(mutant_material, block_def)
        else:
            mutate_methods.mutate_single_argindex(mutant_material, block_def)
