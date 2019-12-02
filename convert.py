import numpy as np
import individual
import problem
# convert from npy to txt file
# YOU MUST CHANGE THE TXT FILE STRING AND SPECIFY WHICH GEN TO WRITE IN OUTPUTS
# OTHERWISE IT WILL OVERWRITE!!!!!
#
# THIS IS NOT POLISHED CODE!!!!!!
def convert(individuals):
    s = ''
    for ind_1 in individuals:
        ind_1 = individual.build_individual(problem.skeleton_genome, ind_1)
        print("fitness", ind_1.fitness.values)
        for i in range(1,ind_1.num_blocks+1):
            curr_block = ind_1.skeleton[i]["block_object"]
            for active_node in curr_block.active_nodes:
                print(curr_block[active_node])
                s += str(curr_block[active_node])
    return s

data = np.load('outputs_cifar/gen2_pop.npy', allow_pickle=True)
=======
data = np.load('outputs_cifar/gen1_pop.npy', allow_pickle=True)
>>>>>>> 85efd119b0aaf581e079cfbff678092f818a53fe
s = convert(data)
text_file = open("gen1_pop.txt", "w")
text_file.write(s)
