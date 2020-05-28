'''
root/codes/individual_definitions/individual_mate.py

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
from codes.genetic_material import IndividualMaterial
from codes.individual_definitions.individual_definition import IndividualDefinition



class IndividualMate_Abstract(ABC):
    '''
    REQUIREMENTS/EXPECTATIONS
    Individual Mate class:
     * if a block is mated, need_evaluate should be set to True at this level no matter what
     * there is a wide variation of ways we can mate so deepcopies should occur at the mate_methods level, not here or block
     * inputs: instance of IndividualDefinition and then two instances of IndividualMaterial as the parents 
     * returns: a list of new offspring individuals or an empty list
    '''
    def __init__(self):
        pass


    @abstractmethod
    def mate(self, parent1: IndividualMaterial, parent2: IndividualMaterial, indiv_def: IndividualDefinition):
        pass


    def set_need_evaluate(self, child: IndividualMaterial, mated_block_index: int):
        for block_index, block_material in enumerate(child.blocks):
            if block_index >= mated_block_index:
                block_material.need_evaluate = True



class IndividualMate_RollOnEachBlock(IndividualMate_Abstract):
    '''
    words
    '''
    def __init__(self):
        pass


    def mate(self, parent1: IndividualMaterial, parent2: IndividualMaterial, indiv_def: IndividualDefinition):
        all_children = []
        for block_index, block_def in enumerate(indiv_def.blocks):
            roll = rnd.random()
            if roll < block_def.mate_def.prob_mate:
                children = block_def.mate(parent1, parent2, block_index)
                # for each child, we need to set need_evaluate on all nodes from the mated block and on
                for child in children:
                    self.set_need_evaluate(child, block_index)
                    all_children.append(child)
        return all_children