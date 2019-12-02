import numpy as np
import individual

# convert from npy to txt file
# YOU MUST CHANGE THE TXT FILE STRING AND SPECIFY WHICH GEN TO WRITE IN OUTPUTS
# OTHERWISE IT WILL OVERWRITE!!!!!
#
# THIS IS NOT POLISHED CODE!!!!!!

def convert(individuals):
    s = ''
    for ind_1 in individuals:
        for i in range(1,ind_1.num_blocks+1):
            curr_block = ind_1.skeleton[i]["block_object"]
            for active_node in curr_block.active_nodes:
                print(curr_block[active_node])
                s += curr_block[active_node]

    return s

data = np.load('outputs_cifar/gen0_pop.npy', allow_pickle=True)
s = convert(data)
text_file = open("gen0_pop.txt", "w")
text_file.write(s)
