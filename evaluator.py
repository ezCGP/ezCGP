# script to evaluate individual with best score from latest generation (.npy) in outputs_cifar/
# and run it for many epochs (see Constants)

import problem
import numpy as np
from individual import Individual, build_individual
import time
from matplotlib import pyplot as plt

#  Constants (you edit)
root_dir = problem.SEED_ROOT_DIR
epochs = 30 
print('Picking best individual from {} and running for {} epochs'.format(root_dir, epochs))

file_generation = '{}/generation_number.npy'.format(root_dir)
generation = np.load(file_generation)
#generation = 20

file_pop = '{}/gen{}_pop.npy'.format(root_dir, generation)
population = np.load(file_pop, allow_pickle = True)
population = [build_individual(problem.skeleton_genome, x) for x in population]
scores = []
for individual in population:
    scores.append(individual.fitness.values[0])
sample_best = population[np.random.choice(a=np.where(np.min(scores) == scores)[0], size=1)[0]] #  choose ind with the best score
#sample_best = population[0] # choose individual manually

#  Copied from tester.py
print("Best Individual from generation {}".format(generation))
individual = sample_best
print('Original fitness: {} \n'.format(individual.fitness.values))
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

# Concatenate validation set and testing set
x_test = problem.x_test[0]
y_test = problem.y_test[0]

x_train = np.array([np.vstack((problem.x_train[0], problem.x_val))])
y_train = np.append(problem.y_train, problem.y_val)

# get the accuracy for each epoch
accuracies = []
f1_scores = []
for epoch in range(epochs):
    curr_block.n_epochs = epoch

    start = time.time()
    individual.evaluate(x_train, y_train, (x_test, y_test)) # This will force genome_output_values to point to the test set
    print("Time to evaluate", time.time() - start)
    individual.fitness.values = problem.scoreFunction(actual=y_test, predict=individual.genome_outputs)
    accuracies.append(individual.fitness.values[0])
    f1_scores.append(individual.fitness.values[1])
    print("{} epoch fitness: {}".format(epoch, individual.fitness.values))

# plot epochs vs. accuracy
plt.scatter(range(epochs), accuracies)
plt.xlabel('Epochs')
plt.ylabel('1 - Accuracy')
plt.title('Generation {} Best Individual'.format(generation))
plt.savefig('{}/gen{}_epochs.png'.format(problem.SEED_ROOT_DIR, generation))

# print epoch accuracy tuples
print('\nFinal Results')
for i in range(epochs):
    print('{} epoch fitness: {}'.format(i, accuracies[i]))
