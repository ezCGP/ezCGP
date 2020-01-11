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
            # show block name
            nickname = ind_1.skeleton[i]['nickname']
            s += '\n\n{} Block:\n'.format(nickname)

            # get block object from block dictionary containing metadata + block obj
            curr_block = ind_1.skeleton[i]['block_object']
            arg_values = np.array(curr_block.args)

            # go through each active genome node and print
            for active_node in curr_block.active_nodes[:-1]:
                # print all layers except last node because it is just a number
                # i'm not sure why, but it isn't a layer type so it should be ignored
                fn = curr_block[active_node]
                if active_node < 0:
                    # nothing to evaluate at input nodes
                    s += 'function at: {} is: {}'\
                              .format(active_node, fn)
                    continue
                elif active_node >= curr_block.genome_main_count:
                    # nothing to evaluate at output nodes
                    s += 'function at: {} is: {} -> likely an output node'\
                              .format(active_node, fn)
                    continue
                s += '\nfunction at: {} is: {} and has arguments: {}'.format(active_node, fn, arg_values[fn['args']])

    print(s) # print total file output
    return s

try:
    print('Converting {}/{} to text'.format(SEED_ROOT_DIR, sys.argv[1]))
    data = np.load('{}/{}'.format(SEED_ROOT_DIR, sys.argv[1]), allow_pickle=True)
except IndexError:
    print('Please input a gen .npy file from outputs_cifar')
    quit()
s = convert(data)
#text_file = open("gen6_pop.txt", "w")
#text_file.write(s)
