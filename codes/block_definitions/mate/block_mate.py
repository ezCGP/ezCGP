'''
root/codes/block_definitions/mate/block_mate.py

Overview:
Block level definition on how to mate 2 parent blocks. We do define an ABC with a mate method that takes in full IndividualMaterial instances (rather than BlockMaterials); in first developping, it seemed easier to send in the full individual genome because while it was clear how the parent's would exchange the block genome, it was not clear how the child would populate itself in the other blocks. So in these mate() methods, we force the user to define how it will perform crossover on the block but also how it will distribute the rest of the blocks to the child.
Also should be noted that all the copying/deepcopying of the parents will happen at the utilities/mate_methods.py level; not here. Expect to return a list of full individual children.

Rules:
Pretty much just need to remember to set a prob_mate attribute, and make sure that the mate method we create has the same inputs:
* 2 parent IndividualMaterial instances
* the BlockDefinition and the block_index
'''

### packages
from abc import ABC, abstractmethod

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.genetic_material import IndividualMaterial
#from codes.block_definitions.block_definition import BlockDefinition #circular dependecy
from codes.block_definitions.utilities import mate_methods
from codes.utilities.custom_logging import ezLogging



class BlockMate_Abstract(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def mate(self,
             parent1: IndividualMaterial,
             parent2: IndividualMaterial,
             block_def,#: BlockDefinition,
             block_index: int):
        pass



class BlockMate_WholeOnly(BlockMate_Abstract):
    '''
    each pair of block/parents will mate w/prob 25%
    if they mate, they will only mate with whole_block()
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockMate_WholeOnly Class" % (None, None))
        self.prob_mate = 1.0

    def mate(self,
             parent1: IndividualMaterial,
             parent2: IndividualMaterial,
             block_def,#: BlockDefinition,
             block_index: int):
        ezLogging.info("%s+%s-%s - Sending %i block to mate_methods.whole_block()" % (parent1.id, parent2.id, block_def.nickname, block_index))
    	# dont actually need block_def
        return mate_methods.whole_block(parent1, parent2, block_index)


class BlockMate_WholeOnly_4Blocks(BlockMate_WholeOnly):
    '''
    change prob by 1/numblocks
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockMate_WholeOnly Class" % (None, None))
        self.prob_mate = 0.33


class BlockMate_SinglePoint(BlockMate_Abstract):
    '''
    each pair of block/parents will mate w/prob 25%
    if they mate, they will only mate with whole_block()
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockMate_SinglePoint Class" % (None, None))
        self.prob_mate = 1.0

    def mate(self,
             parent1: IndividualMaterial,
             parent2: IndividualMaterial,
             block_def,#: BlockDefinition,
             block_index: int):
        ezLogging.info("%s+%s-%s - Sending %i block to mate_methods.whole_block()" % (parent1.id, parent2.id, block_def.nickname, block_index))
    	# dont actually need block_def
        return mate_methods.one_point_crossover(parent1, parent2, block_index)



class BlockMate_NoMate(BlockMate_Abstract):
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockMate_NoMate Class" % (None, None))
        self.prob_mate = 0

    def mate(self,
             parent1: IndividualMaterial,
             parent2: IndividualMaterial,
             block_def, #: BlockDefinition,
             block_index: int):
        ezLogging.info("%s+%s-%s - No mating for block %i" % (parent1.id, parent2.id, block_def.nickname, block_index))
        return []
