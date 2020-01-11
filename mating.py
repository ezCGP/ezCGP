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
        block_index = 2
        while block_index == 2:
            # if it picked the unchanging preprocessing block (ceil grayscale), then pick another block.
            # this loop should be deleted ONLY if that ceil grayscale block gets more primitives later
            block_index = np.random.choice(a=range(1, parent_1.num_blocks+1), size=1, p=None)[0]

        print('block index: ', block_index)

        # swap the blocks, genome_output_values already clear -> no mem leak in deepcopy
        # genome_list is like [block1, args1, block2, args2, ...], see build_individual()
        # swap the block itself
        temp_block = deepcopy(ind_2_genome_list[2*block_index-2])
        ind_2_genome_list[2*block_index-2] = deepcopy(ind_1_genome_list[2*block_index-2])
        ind_1_genome_list[2*block_index-2] = deepcopy(temp_block)
        #print('mated block: ', temp_block)

        # swap the block's args
        temp_args = deepcopy(ind_2_genome_list[2*block_index-1])
        ind_2_genome_list[2*block_index-1] = deepcopy(ind_1_genome_list[2*block_index-1])
        ind_1_genome_list[2*block_index-1] = deepcopy(temp_args)
        #print('mated block args: ', temp_args)

        # build the new individuals from the genome lists
        ind_1 = build_individual(self.skeleton_genome, ind_1_genome_list)
        ind_2 = build_individual(self.skeleton_genome, ind_2_genome_list)

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
