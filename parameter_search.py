'''
root/prameter_search.py

Overview:
Want a way to slightly alter our problem class and run the universe on each problem permutation.
Will be used for search of evolutionary parameters like genome size, population size, etc

Rules:
Gonna have to be real careful that we will know which output has which set up parameters. Don't change problem too much!
'''

### packages
import os
import shutil
import time
import numpy as np
import random
import tempfile
import logging
import gc
from mpi4py import MPI

### sys relative to root AND to problem dir to import respective problem file
import sys
from os.path import dirname, realpath, join
sys.path.append(dirname(realpath(__file__)))
sys.path.append(join(dirname(realpath(__file__)), "problems"))

### absolute imports wrt root
from codes.utilities.custom_logging import ezLogging
import main


def parameter_search(problem,
                     log_formatter,
                     problem_output_directory):
    '''
    words
    '''
    node_rank = MPI.COMM_WORLD.Get_rank() # which node are we on if mpi, else always 0
    node_size = MPI.COMM_WORLD.Get_size() # how many nodes are we using if mpi, else always 1

    # TODO!

    # dirty example:
    for genome_size in [2**4, 2**5, 2**6]:
        for pop_size in [50, 100, 200, 500]:
            problem.indiv_def[0].main_count = genome_size
            problem.indiv_def[0].genome_count = genome_size + problem.indiv_def[0].input_count + problem.indiv_def[0].output_count
            problem.indiv_def[0].meta_def.main_count = genome_size
            problem.indiv_def[0].meta_def.genome_count = problem.indiv_def[0].genome_count
            problem.population_size = pop_size

            # GO!
            main.main(problem, log_formatter, problem_output_directory)



if __name__ == "__main__":
    '''
    copy paste from main.py
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--problem",
                        type = str,
                        required = True,
                        help = "pick which problem class to import")
    parser.add_argument("-s", "--seed",
                        type = int,
                        required = False,
                        help = "pick which seed to use for numpy. If not provided, will generate from time.")
    parser.add_argument("-n", "--name",
                        type = str,
                        required = False,
                        default = None,
                        help = "add str to end of output directory name to help with documenting the reason for the run")
    parser.add_argument("-t", "--testing",
                        action = "store_const",
                        const = True,
                        default = False,
                        help = "set flag to document the output folder with 'test' to distinguish it from serious runs")
    parser.add_argument("-d", "--debug",
                        help = "set the logging level to the lowest level to collect everything",
                        dest = "loglevel",
                        action = "store_const",
                        const = logging.DEBUG,
                        default = logging.WARNING)
    parser.add_argument("-v", "--verbose",
                        help = "set the logging level to 2nd lowest level to collect everything except debug",
                        dest = "loglevel",
                        action = "store_const",
                        const = logging.INFO)
    args = parser.parse_args()

    # create a logging directory specifically for this run
    # will be named: root/outputs/problem_file/datetime_as_str/
    time_str = time.strftime("%Y%m%d-%H%M%S")

    # set seed
    if args.seed is None:
        seed = int(time_str.replace("-","")) # from "20200623-101442" to 20200623101442
    else:
        seed = int(args.seed)
    seed%=(2**32) #<-np.random.seed must be between [0,2**32-1]
    
    if args.testing:
        time_str = "testing-%s" % time_str
    problem_output_directory = os.path.join(dirname(realpath(__file__)),
                                            "outputs",
                                            args.problem,
                                            time_str)

    if args.name:
        name_str = "-" + str(args.name).replace(" ", "_")
        problem_output_directory += name_str

    # figure out which problem py file to import
    if args.problem.endswith('.py'):
        problem_filename = os.path.basename(args.problem)
    else:
        problem_filename = os.path.basename(args.problem + ".py")
    
    # RUN BABYYY
    problem, log_formatter = main.setup(problem_filename, problem_output_directory, seed, args.loglevel)
    parameter_search(problem, log_formatter, problem_output_directory)