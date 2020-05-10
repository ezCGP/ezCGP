'''
root/codes/block_definitions/utilities/mate_methods.py

Overview:
attach all mate methods to the MateMethods class

required inputs:
 * 2 parent IndividualMaterial
 * 1 int for the block index we want to mate

required output:
 * it is expected that the parents do not get altered so deepcopies are required
 * list of offspring IndividualMaterial to be added to the population


Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
from copy import deepcopy

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.genetic_material import IndividualMaterial
from codes.block_definitions.block_definition import BlockDefinition


def whole_block(parent1: IndividualMaterial, parent2: IndividualMaterial, block_index: int):
    '''
    TODO
    '''
    child1 = deepcopy(parent1)
    child1[block_index] = deepcopy(parent2[block_index])

    child2 = deepcopy(parent2)
    child2[block_index] = deepcopy(parent1[block_index])

    return [child1, child2]


def partial_block(parent1: IndividualMaterial, parent2: IndividualMaterial, block_def: BlockDefinition, block_index: int):
    '''
    TODO
    '''
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)
    # TODO
    #return [child1, child2]
    return []