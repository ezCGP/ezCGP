'''
root/main.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
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
# moved most imports to AFTER seed is set!


def main(problem_filename: str,
         problem_output_directory: str=tempfile.mkdtemp(),
         seed: int=0,
         loglevel: int=logging.WARNING):
    node_rank = MPI.COMM_WORLD.Get_rank() # which node are we on if mpi, else always 0
    node_size = MPI.COMM_WORLD.Get_size() # how many nodes are we using if mpi, else always 1

    # want to make sure that creation of files, folders, and logging is not duplicated if using mpi.
    # the following is always true if not using mpi 
    if node_rank == 0:
        os.makedirs(problem_output_directory, exist_ok=False)
        # copy problem file over to problem_output_directory
        # this way we know for sure which version of the problem file resulted in the output
        src = join(dirname(realpath(__file__)), "problems", problem_filename)
        dst = join(problem_output_directory, problem_filename)
        shutil.copyfile(src, dst)

    # create custom logging.logger for this node
    log_formatter = ezLogging.logging_setup(loglevel)
    if loglevel < logging.WARNING:
        # true only for DEBUG or INFO
        log_handler_2stdout = ezLogging.logging_2stdout(log_formatter)
    else:
        log_handler_2stdout = None
    
    # set the seed before importing problem.
    # NOTE will set another seed when we start the universe
    ezLogging.warning("Setting seed, for file imports, to %i" % (seed))
    np.random.seed(seed)
    random.seed(seed) # shouldn't be using 'random' module but setting seed jic
    problem_module = __import__(problem_filename[:-3]) #remoe the '.py' from filename
    problem = problem_module.Problem()
    from codes.universe import UniverseDefinition, MPIUniverseDefinition

    log_handler_2file = None # just initializing
    for ith_universe in range(problem.number_universe):
        # set new output directory
        universe_output_directory = os.path.join(problem_output_directory, "univ%04d" % ith_universe)
        if node_rank == 0:
            os.makedirs(universe_output_directory, exist_ok=False)
        MPI.COMM_WORLD.Barrier()

        # init corresponding universe and new log file handler
        if problem.mpi:
            ezLogging_method = ezLogging.logging_2file_mpi
            universe_seed = seed + 1 + (ith_universe*node_size) + node_rank
            ThisUniverse = MPIUniverseDefinition
        else:
            ezLogging_method = ezLogging.logging_2file
            universe_seed = seed + 1 + ith_universe
            ThisUniverse = UniverseDefinition
        log_handler_2file = ezLogging_method(log_formatter, filename=os.path.join(universe_output_directory, "log.txt"))
        ezLogging.log_git_metadata()
        import pdb; pdb.set_trace()
        ezLogging.warning("Setting seed for Universe, to %i" % (universe_seed))
        np.random.seed(universe_seed)
        random.seed(seed)
        ezLogging.warning("STARTING UNIVERSE %i" % ith_universe)
        universe = ThisUniverse(problem, universe_output_directory)

        # run
        start_time = time.time()
        universe.run(problem)
        ezLogging.warning("...time of universe %i: %.2f minutes" % (ith_universe, (time.time()-start_time)/60))
        
        # do some clean up, if we're about to start another run
        # remove previous universe log file handler if exists
        ezLogging.logging_remove_handler(log_handler_2file)
        # TODO is there a way to track memory usage before and after here?
        del universe # will that also delete populations? or at least gc.collect will remove it?
        gc.collect()


if __name__ == "__main__":
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
    main(problem_filename, problem_output_directory, seed, args.loglevel)