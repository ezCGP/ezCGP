import numbers
import gc
import numpy as np
from copy import deepcopy

class Mate():
    def __init__(self, pop, skeleton_genome):
        self.pop = pop
        self.skeleton_genome = skeleton_genome

    def whole_block_swapping(self):
        # select two random individuals
        item1, item2 = np.random.choice(a=self.pop, size=2, p=None)
        # TODO: CLEAR OUT DEEPCOPY genome_output_values of the blocks
        ind_1 = deepcopy(item1)
        ind_2 = deepcopy(item2)
        # select two random blocks in the individuals
        block_indices = [key for key in self.skeleton_genome.keys() if isinstance(key, numbers.Number)]
        block_index = np.random.choice(a=block_indices, size=1, p=None)[0]

        # TODO: CHECK FOR MEMORY LEASKS HERE!
        temp_block = deepcopy(ind_1.skeleton[block_index]['block_object'])
        ind_1.skeleton[block_index]['block_object'] = deepcopy(ind_2.skeleton[block_index]['block_object'])
        ind_2.skeleton[block_index]['block_object'] = deepcopy(temp_block)
        ind_1.skeleton[block_index]['block_object'].need_evaluate = True
        ind_2.skeleton[block_index]['block_object'].need_evaluate = True
        del temp_block
        gc.collect()

        return [ind_1, ind_2]