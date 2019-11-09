### main.py
import utils.LogSetup as logging
# external packages
from copy import deepcopy
import numpy as np
import random
import time
import logging

if __name__ == '__main__':
    # set the seed and import scripts
    seed = 5
    np.random.seed(seed)
    random.seed(seed) #set both random seeds to same thin
    # keep these imports after the seed is set for numpy
    import problem
    import universe

    # Read in Data
    train_data = problem.x_train
    train_labels = problem.y_train

    final_populations = [] # one for each universe created
    num_universes = 1#20
    for i in range(num_universes):
        logging.info('start new run {}'.format(i))
        start = time.time()
        converged_solution = universe.create_universe(input_data=train_data, labels=train_labels, universe_seed=seed+i, population_size=40)
        final_populations.append(converged_solution)
        time_complete = time.time() - start
        logging.info('time of generation: {}'.format(time.time() - start))
        with open("sequential_run_time.txt", "a") as f:
            f.write("%f\n" % time_complete)
        # post processing step for that run
