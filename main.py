'''
root/main.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
import os
import time
import numpy as np
import tempfile

### sys relative to root AND to problem dir to import respective problem file
import sys
from os.path import dirname, realpath, join
sys.path.append(dirname(realpath(__file__)))
sys.path.append(join(dirname(realpath(__file__)), "problems"))

### absolute imports wrt root
from codes.universe import UniverseDefinition, MPIUniverseDefinition
from problem.problem_abstract import ProblemDefinition_Abstract


def main(problem: ProblemDefinition_Abstract
         probelm_output_directory=tempfile.mkdtemp(),
         seed: int=0):
    
    for ith_universe in range(problem.number_universe):
        # set the seed
        np.random.seed(seed + ith_universe)

        # init corresponding universe
        universe_output_direcotry = os.path.join(probelm_output_directory, "univ%i" % ith_universe)
        if problem.mpi:
            universe = MPIUniverseDefinition(problem, universe_output_direcotry)
        else:
            universe = UniverseDefinition(problem, universe_output_direcotry)

        # run
        start_time = time.time()
        universe.run(problem)
        print("time of universe %i: %02fmin" % (ith_universe, (time.time()-start_time)/60))


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

    # create a logging directory specifically for this run
    # will be named: root/outputs/problem_file/datetime_as_str/
    probelm_output_directory = os.path.join(dirname(realpath(__file__)),
                                                "outputs",
                                                args.problem,
                                                time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(probelm_output_directory, exist_ok=False)

    # figure out which problem py file to import
    if args.problem.endswith('.py'):
        args.problem = args.problem[:-3]
    problem_module = __import__(args.problem)
    problem = problem_module.Problem()
    
    # RUN BABYYY
    main(problem, probelm_output_directory, args.seed)