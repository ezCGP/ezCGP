'''
root/codes/block_definitions/block_mate.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
from abc import ABC, abstractmethod

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from codes.genetic_material import IndividualMaterial
from codes.block_definitions.block_definition import BlockDefinition
from codes.block_defintions.utilities import mate_methods



class BlockMate_Abstract(ABC):
    '''
    REQUIREMENTS/EXPECTATIONS
    Block Mate class:
     * in __init__ will assign a prob_mate attribute for that block
     * as above, we should not deepcopy at all here; we assume that the mate_method itself will handle that and simply return the list
        output by the select mate_method
     * inputs: the 2 parents as instances of IndividualMaterial, integer for the i^th block we want to mate
     * returns: a list of offspring output by the selected mate_method
    '''
    def __init__(self):
        pass

    @abstractmethod
    def mate(self, parent1: IndividualMaterial, parent2: IndividualMaterial, block_def: BlockDefinition, block_index: int):
        pass



class BlockMate_WholeOnly(BlockMate_Abstract):
    '''
    each pair of block/parents will mate w/prob 25%
    if they mate, they will only mate with whole_block()
    '''
    def __init__(self):
        self.prob_mate = 1.0

    def mate(self, parent1: IndividualMaterial, parent2: IndividualMaterial, block_def: BlockDefinition, block_index: int):
    	# dont actually need block_def
        return mate_methods.whole_block(parent1, parent2, block_index)



class BlockMate_NoMate(BlockMate_Abstract):
    def __init__(self):
        self.prob_mate = 0

    def mate(self, parent1: IndividualMaterial, parent2: IndividualMaterial, block_def: BlockDefinition, block_index: int):
        return []