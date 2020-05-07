'''

'''

# packages
import os
import time
import numpy as np


# scripts
from code.universe import Universe, MPIUniverse


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--problem",
                        type = str,
                        required = True,
                        help = "pick which problem class to import")
    parser.add_argument("-s", "--seed",
                        type = int,
                        default = 0,
                        help = "pick which seed to use for numpy")
    args = parser.parse_args()



    # figure out which problem py file to import
    if args.problem.endswith('.py'):
        args.problem = args.problem[:-3]
    problem_module = __import__(args.problem)


    output_home = os.path.join(os.getcwd(),
                               "outputs",
                               args.problem,
                               time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_home, exist_ok=False)

    problem = problem_module.Problem()
    for ith_universe in range(problem.number_universe):
        # set the seed
        np.random.seed(args.seed + ith_universe)

        # init corresponding universe
        output_universe = os.path.join(output_home, "univ%i" % ith_universe)
        if problem.mpi:
            universe = MPIUniverse(problem, output_universe)
        else:
            universe = Universe(problem, output_universe)

        # run
        start_time = time.time()
        universe.run(problem)
        print("time of universe %i: %02fmin" % (ith_universe, (time.time()-start_time)/60))