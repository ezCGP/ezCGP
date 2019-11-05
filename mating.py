import numbers
import gc
import numpy as np
from copy import deepcopy

class Mate():
    def __init__(self, population, skeleton_genome):
        self.population = population
        self.skeleton_genome = skeleton_genome

    def whole_block_swapping(self):
        # select two random individuals
        parent_1, parent_2 = np.random.choice(a=self.population, size=2, p=None)
        
        # copy paernts to new individuals before swapping
        ind_1 = deepcopy(parent_1)
        ind_2 = deepcopy(parent_2)

        # clear out both new individuals' blocks' genome_outputs to avoid memory leak
        # genome_outputs are typically large, esp after training/evaluation
        for block_index in ind_1.blocks_indices:
            ind_1.skeleton[block_index]['block_object'].genome_output_values = []

        for block_index in ind_2.blocks_indices:
            ind_2.skeleton[block_index]['block_object'].genome_output_values = []

        # select two random blocks in the individuals
        block_index = np.random.choice(a=ind_1.blocks_indices, size=1, p=None)[0]

        # swap the blocks, genome_output_values already clear -> no mem leak in deepcopy
        temp_block = deepcopy(ind_1.skeleton[block_index]['block_object'])
        ind_1.skeleton[block_index]['block_object'] = deepcopy(ind_2.skeleton[block_index]['block_object'])
        ind_2.skeleton[block_index]['block_object'] = deepcopy(temp_block)
        ind_1.skeleton[block_index]['block_object'].need_evaluate = True
        ind_2.skeleton[block_index]['block_object'].need_evaluate = True
        del temp_block
        gc.collect()

        return [ind_1, ind_2]
