import sys
import numpy as np
import individual
from problem import skeleton_genome, SEED_ROOT_DIR
# convert from npy to txt file and shows blocks and their layers
#
# USAGE
#
# python3 convert.py <file_name>
#
# where <file_name> is the name of the .npy file
#
# e.g. python3 convert.py gen4_pop.npy

def convert(individuals):
    s = ''
    for i, ind_1 in enumerate(individuals):
        s += '\n\nIndividual number {}:'.format(i)
        ind_1 = individual.build_individual(skeleton_genome, ind_1)
        s += "\nfitness {}".format(ind_1.fitness.values)
        for i in range(1,ind_1.num_blocks+1):
            # get block dictionary containing metadata + block obj
            curr_block = ind_1.skeleton[i]

            # show block name
            s += '\n\n{} Block:'.format(curr_block['nickname'])

            # go through each active genome node and print
            for active_node in curr_block['block_object'].active_nodes[:-1]:
                # print all layers except last node because it is just a number
                # i'm not sure why, but it isn't a layer type so it should probably be ignored
                s += '\n' + str(curr_block['block_object'][active_node])

    print(s) # print total file output
    return s

try:
    data = np.load('{}/{}'.format(problem.SEED_ROOT_DIR, sys.argv[1]), allow_pickle=True)
except IndexError:
    print('Please input a gen .npy file from outputs_cifar')
    quit()
s = convert(data)
#text_file = open("gen6_pop.txt", "w")
#text_file.write(s)
