### main.py

# external packages
from copy import deepcopy

import problem

# my scripts
import universe
num_universes = 1

if __name__ == '__main__':
    # Read in Data
    train_data = problem.x_train
    train_labels = problem.y_train
    seed = 10

    final_populations = [] # one for each universe created
    for i in range(num_universes):
        print("start new run %i" % i)
        converged_solution = universe.create_universe(input_data=train_data, labels=train_labels, universe_seed=seed+i)
        final_populations.append(converged_solution)

        # post processing step for that run
