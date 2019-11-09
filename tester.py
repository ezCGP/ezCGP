import utils.LogSetup #import this first
from individual import Individual, create_individual_from_genome_list
import problem
import numpy as np
train_data = problem.x_train
train_labels = problem.y_train

individual = Individual(problem.skeleton_genome)
individual.evaluate(problem.x_train, problem.y_train, (problem.x_val, problem.y_val))

individual.fitness.values = problem.scoreFunction(actual=problem.y_val, predict=individual.genome_outputs)
print('individual has fitness: {}'.format(individual.fitness.values))

# print(individual.genome_outputs)
# print(individual.dead)

# for i in range(1,individual.num_blocks+1):
#     curr_block = individual.skeleton[i]["block_object"]
#     # print('curr_block: {}'.format(curr_block))
#     arg_values = np.array(curr_block.args)
#     print('arg_values: {}'.format(arg_values))
#     print('curr_block isDead = ', curr_block.dead)
#     print(curr_block.active_nodes)
#     for active_node in curr_block.active_nodes:
#         fn = curr_block[active_node]
#         if active_node < 0:
#             # nothing to evaluate at input nodes
#             print('function at: {} is: {}'\
#                 .format(active_node, fn))
#             continue
#         elif active_node >= curr_block.genome_main_count:
#             # nothing to evaluate at output nodes
#             print('function at: {} is: {} -> likely an output node'\
#                 .format(active_node, fn))
#             continue
#         print('function at: {} is: {} and has arguments: {}'\
#                 .format(active_node, fn, arg_values[fn['args']]))
