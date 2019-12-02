import problem
import numpy as np
from individual import Individual, build_individual

#  Constants (you edit)
root_dir = 'outputs_cifar'
epochs = 2

file_generation = '{}/generation_number.npy'.format(root_dir)
generation = np.load(file_generation)

file_pop = '{}/gen{}_pop.npy'.format(root_dir, generation)
population = np.load(file_pop, allow_pickle = True)
population = [build_individual(problem.skeleton_genome, x) for x in population]
scores = []
for individual in population:
    scores.append(individual.fitness.values[0])
sample_best = population[np.random.choice(a=np.where(np.min(scores) == scores)[0], size=1)[0]] #  choose ind with the best score

#  Copied from tester.py
print("Best Individual")
individual = sample_best
for i in range(1,individual.num_blocks+1):
    curr_block = individual.skeleton[i]["block_object"]
    # print('curr_block: {}'.format(curr_block))
    arg_values = np.array(curr_block.args)
    print('arg_values: {}'.format(arg_values))
    print('curr_block isDead = ', curr_block.dead)
    print(curr_block.active_nodes)
    for active_node in curr_block.active_nodes:
        fn = curr_block[active_node]
        if active_node < 0:
            # nothing to evaluate at input nodes
            print('function at: {} is: {}'\
                .format(active_node, fn))
            continue
        elif active_node >= curr_block.genome_main_count:
            # nothing to evaluate at output nodes
            print('function at: {} is: {} -> likely an output node'\
                .format(active_node, fn))
            continue
        print('function at: {} is: {} and has arguments: {}'\
                .format(active_node, fn, arg_values[fn['args']]))

curr_block.n_epochs = epochs

# Concatenate validation set and testing set
x_test = np.vstack((problem.x_test[0], problem.x_val))
y_test = np.append(problem.y_test[0], problem.y_val)
x_train = problem.x_train
y_train = problem.y_train


individual.evaluate(x_train, y_train, (x_test, y_test)) # This will force genome_output_values to point to the test set
individual.fitness.values = problem.scoreFunction(actual=y_test, predict=individual.genome_outputs)

print("final fitness", individual.fitness.values)