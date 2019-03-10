### main.py

# external packages
from copy import deepcopy
import numpy as np


if __name__ == '__main__':
    # set the seed and import scripts
    seed = 4
    np.random.seed(seed)
    import problem
    import universe

    # Read in Data
    train_data = problem.x_train
    train_labels = problem.y_train

    final_populations = [] # one for each universe created
    num_universes = 1#20
    for i in range(num_universes):
        print("start new run %i" % i)
        converged_solution = universe.create_universe(input_data=train_data, labels=train_labels, universe_seed=seed+i)
        final_populations.append(converged_solution)

        # post processing step for that run
