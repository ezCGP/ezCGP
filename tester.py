from individual import Individual
import problem
import numpy as np

train_data = problem.x_train
train_labels = problem.y_train

individual = Individual(problem.skeleton_genome)
individual.evaluate(problem.x_train, problem.y_train, (problem.x_val, problem.y_val))
# print(individual.genome_outputs)
# print(individual.dead)
for i in range(1,individual.num_blocks+1):
    curr_block = individual.skeleton[i]["block_object"]
    arg_values = np.array(curr_block.args)
    print('arg_values: {}'.format(arg_values))
    print('curr_block isDead = ', curr_block.dead)
    print(curr_block.active_nodes)
    for active_node in curr_block.active_nodes:
        fn = curr_block[active_node]
        # print(type(fn))
        # print('fn[args]: {}'.format(fn['args']))
        print('function at: {} is: {}'\
            .format(active_node, fn))
        # print('Args are: ')

individual.mutate()
for i in range(1,individual.num_blocks+1):
    curr_block = individual.skeleton[i]["block_object"]
    print('curr_block isDead = ', curr_block.dead)
    arg_values = np.array(curr_block.args)
    print('arg_values: {}'.format(arg_values))
    print(curr_block.active_nodes)
    for active_node in curr_block.active_nodes:
        fn = curr_block[active_node]
        # print(fn)
        print('function at: {} is: {}'.format(active_node, fn))

# print('labels: {} have shape: {}\ngenome_outputs: {} have shape: {}'\
#     .format(train_labels, train_labels.shape, individual.genome_outputs, individual.genome_outputs.shape))
# individual.fitness.values = problem.scoreFunction(actual=problem.y_val, predict=individual.genome_outputs)
# print('individual has fitness: ', individual.fitness.values)
