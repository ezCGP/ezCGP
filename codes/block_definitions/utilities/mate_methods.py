'''
root/codes/block_definitions/utilities/mate_methods.py

Overview:
attach all mate methods to the MateMethods class

required inputs:
 * 2 parent IndividualMaterial
 * block def
 * 1 int for the block index we want to mate

required output:
 * it is expected that the parents do not get altered so deepcopies are required
 * list of offspring IndividualMaterial to be added to the population
'''

### packages
from copy import deepcopy

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.genetic_material import IndividualMaterial
#from codes.block_definitions.block_definition import BlockDefinition #circular dependecy
from codes.utilities.custom_logging import ezLogging


def whole_block(parent1: IndividualMaterial,
                parent2: IndividualMaterial,
                block_index: int):
    '''
    Super simple direct swaping of the blocks. 2 parents in; 2 children out.
    '''
    ezLogging.info("%s+%s - Mating Block %i with whole_block()" % (parent1.id, parent2.id, block_index))
    child1 = deepcopy(parent1)
    child1[block_index] = deepcopy(parent2[block_index])

    child2 = deepcopy(parent2)
    child2[block_index] = deepcopy(parent1[block_index])

    return [child1, child2]


def partial_block(parent1: IndividualMaterial,
                  parent2: IndividualMaterial,
                  block_def, #: BlockDefinition,
                  block_index: int):
    '''
    TODO
    '''
    ezLogging.info("%s+%s - Mating Block %i with partial_block()" % (parent1.id, parent2.id, block_index))
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)
    # TODO
    #return [child1, child2]
    return []


def one_point_crossover(parent1: IndividualMaterial,
                        parent2: IndividualMaterial,
                        block_def, #: BlockDefinition,
                        block_index: int):
    '''
    Vishesh's Task

    Attempt to implement One-Point AKA Single-Point Crossover

    Assumptions we'll be making:
        * TODO
    '''
    ezLogging.info("%s+%s - Mating Block %i with partial_block()" % (parent1.id, parent2.id, block_index))
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)


    # TODO!
    


    return [child1, child2]