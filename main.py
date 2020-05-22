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
import logging
import gc

### sys relative to root AND to problem dir to import respective problem file
import sys
from os.path import dirname, realpath, join
sys.path.append(dirname(realpath(__file__)))
sys.path.append(join(dirname(realpath(__file__)), "problems"))

### absolute imports wrt root
from codes.universe import UniverseDefinition, MPIUniverseDefinition
from problem.problem_abstract import ProblemDefinition_Abstract


def main(problem: ProblemDefinition_Abstract,
         probelm_output_directory=tempfile.mkdtemp(),
         seed: int=0,
         loglevel=logging.WARNING):
    # init logging here but then set the filepath inside the for-loop so that
    # we have a unique log file per universe run
    logging.basicConfig(level=loglevel) #not sure if i can set format=log_formatter here to be true for all handlers
    log_logger = logging.getLogger()
    log_formatter = logging.Formatter(fmt="%(asctime)s,%(msecs)d \
                                           [%(threadName)s-%(thread)d] - \
                                           [%(filename)s-%funcName)s] - \
                                           %(levelname)s - \
                                           %(message)s",
                                      datefmt="%H:%M:%S")
    if loglevel < logging.WARNING: # true only for DEBUG or INFO
        log_stdouthandler = logging.StreamHandler(sys.stdout)
        log_stdouthandler.setFormatter(log_formatter)
        log_logger.addHandler(log_stdouthandler)
    else:
        log_stdouthandler = None
    
    for ith_universe in range(problem.number_universe):
        # set the seed
        np.random.seed(seed + ith_universe)

        # set new output directory
        universe_output_direcotry = os.path.join(probelm_output_directory, "univ%i" % ith_universe)
        os.makedirs(universe_output_direcotry, exist_ok=False)
        # remove any old logging file handlers
        for old_filehandler in logging.handlers:
            if old_filehandler != log_stdouthandler:
                # remove everything except the stdout one
                logging.getLogger().removeHandler(old_filehandler)
                del old_filehandler
        # replace with a new file handler for the new output directory
        log_filehandler = logging.FileHandler(os.path.join(universe_output_direcotry, "log.txt"), 'w')
        log_filehandler.setFormatter(log_formatter)
        logging.getLogger().addHandler(log_filehandler)
        
        # init corresponding universe
        if problem.mpi:
            universe = MPIUniverseDefinition(problem, universe_output_direcotry)
        else:
            universe = UniverseDefinition(problem, universe_output_direcotry)

        # run
        start_time = time.time()
        universe.run(problem)
        print("time of universe %i: %02fmin" % (ith_universe, (time.time()-start_time)/60))
        
        # do some clean up, if we're about to start another run
        if ith_universe+1 < problem.number_universe:
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
                        default = 0,
                        help = "pick which seed to use for numpy")
    parser.add_argument("-d", "--debug",
                        help="set the logging level to the lowest level to collect everything",
                        dest="loglevel",
                        const=logging.DEBUG,
                        default=logging.WARNING)
    parser.add_argument("-v", "--verbose",
                        help="set the logging level to 2nd lowest level to collect everything except debug",
                        dest="loglevel"
                        const=logging.INFO)
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
    main(problem, probelm_output_directory, args.seed, args.loglevel)