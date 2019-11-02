import number
import gc

class Mate(Genome):
    def __init__(self, pop, skeleton_genome):
        self.pop = pop
        self.skeleton_genome = skeleton_genome

    def whole_block_swapping(self):
        # select two random individuals
        ind_1, ind_2 = np.random.choice(a=pop, size=2, weights=None)
        
        # select two random blocks in the individuals
        block_indices = [key for key in self.skeleton_genome.keys() if isinstance(key, numbers.Number)]
        block_index = np.random.choices(a=block_indices, size=1, weights=None)[0]
        
        # TODO: CHECK FOR MEMORY LEASKS HERE!
        temp_block = deepcopy(ind_1.skeleton[block_index]['block_object'])
        ind_1.skeleton[block_index]['block_object'] = deepcopy(ind_2.skeleton[block_index]['block_object'])
        ind_2.skeleton[block_index]['block_object'] = deepcopy(temp_block)
        del temp_block
        gc.collect()

        return ind_1, ind_2
