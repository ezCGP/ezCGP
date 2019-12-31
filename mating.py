import numbers
import gc
import numpy as np
from copy import deepcopy
from individual import Individual, build_individual

class Mate():
    def __init__(self, population, skeleton_genome):
        self.population = population
        self.skeleton_genome = skeleton_genome

    def whole_block_swapping(self):
        # select two random individuals
        parent_1, parent_2 = np.random.choice(a=self.population, size=2, p=None)

        # print parents
        # print("parent1")
        # for i in range(1,parent_1.num_blocks+1):
        #     curr_block = parent_1.skeleton[i]["block_object"]
        #     for active_node in curr_block.active_nodes:
        #         print(curr_block[active_node])
        # print("parent2")
        # for i in range(1,parent_2.num_blocks+1):
        #     curr_block = parent_2.skeleton[i]["block_object"]
        #     for active_node in curr_block.active_nodes:
        #         print(curr_block[active_node])

        # copy paernts to new individuals before swapping
        ind_1_genome_list = parent_1.get_genome_list()
        ind_2_genome_list = parent_2.get_genome_list()


        # select two random blocks in the individuals
        block_index = 1
        while block_index == 1:
            # if it picked the unchanging preprocessing block (ceil grayscale), then pick another block.
            # this should be deleted ONLY if that ceil grayscale block gets more primitives later
            block_index = np.random.choice(a=parent_1.blocks_indices, size=1, p=None)[0]

        # get genome list is an array, which starts with 0
        block_index -= 1
        print('block index: ', block_index)

        # swap the blocks, genome_output_values already clear -> no mem leak in deepcopy
        temp_block = deepcopy(ind_2_genome_list[block_index])
        ind_2_genome_list[block_index] = deepcopy(ind_1_genome_list[block_index])
        ind_1_genome_list[block_index] = deepcopy(temp_block)
        ind_1 = build_individual(self.skeleton_genome, ind_1_genome_list)
        ind_2 = build_individual(self.skeleton_genome, ind_2_genome_list)

        # accessing dictionary since we're back to object, so add 1 back
        block_index += 1
        ind_1.set_need_evaluate(flag=True) # set all need_evaluate block flags to True
        ind_2.set_need_evaluate(flag=True)
        #ind_1.skeleton[block_index]['block_object'].need_evaluate = True
        #ind_2.skeleton[block_index]['block_object'].need_evaluate = True
        del temp_block
        gc.collect()

        # print out mated individuals
        # print("ind1")
        # for i in range(1,ind_1.num_blocks+1):
        #     curr_block = ind_1.skeleton[i]["block_object"]
        #     for active_node in curr_block.active_nodes:
        #         print(curr_block[active_node])
        # print("ind2")
        # for i in range(1,ind_2.num_blocks+1):
        #     curr_block = ind_2.skeleton[i]["block_object"]
        #     for active_node in curr_block.active_nodes:
        #         print(curr_block[active_node])

        return [ind_1, ind_2]
