'''
Working on a project that wants to have ezCGP 'plugged into' another piece of code
that will handle the scoring, selection, and termination of evolution; let's call
that code the 'regulator'.
My code should just be plugged in to handle the evolution of the population only.
ezCGP needs to be imported into the regulator, a class loaded and instantiated
to handle the evolution and a run method to trigger a single generation of 
evolution.

Strategy:
* make a class to mimic what main.main() does
* make sure there is an __init__ and run method
'''
### packages
import os
import time
import numpy as np
import random
import tempfile
import logging
import gc

### absolute imports
from codes.utilities.custom_logging import ezLogging


class ezCGP_Module():
    def __init__(self, config_filepath, problem_filepath, seed=None):
        '''
        * high level, going to mimic what main.main() does
        * further will try to digest some config file to help
            dynamically set Problem() variables/setting
        '''
        assert(os.path.exists(config_filepath)), "Given config file %s does not exist" % config_filepath
        assert(os.path.exists(problem_filepath)), "Given problem file %s does not exist" % problem_filepath

        # create output directory
        time_str = time.strftime("%Y%m%d-%H%M%S")
        problem_filename = os.path.basename(problem_filepath)
        self.universe_output_directory = os.path.join(dirname(realpath(__file__)),
                                                      "outputs",
                                                      problem_filename,
                                                      time_str)

        # set up logging
        log_handler_2file = log_formatter = ezLogging.logging_setup(logging.WARNING)
        ezLogging.logging_2file(log_formatter, filename=os.path.join(self.universe_output_directory, "log.txt"))


        # set seed
        if seed is None:
            seed = int(time_str.replace("-","")) # from "20200623-101442" to 20200623101442
        else:
            seed = int(seed)
        seed%=(2**32) #<-np.random.seed must be between [0,2**32-1]
        # always set the seed before importing any file from ezCGP
        ezLogging.warning("Setting seed, for file imports, to %i" % (seed))
        random.seed(seed)
        np.random.seed(seed)

        # import problem file
        sys.path.join(os.path.dirname(problem_filepath))
        problem_module = __import__(problem_filename[:-3]) #remove the '.py' from filename
        self.problem = problem_module.Problem(config_filepath)

        # import universe
        from codes.universe import UniverseEvolveOnly
        universe_seed = seed + 1
        ezLogging.warning("Setting seed for Universe, to %i" % (universe_seed))
        random.seed(universe_seed)
        np.random.seed(universe_seed)
        ezLogging.warning("STARTING UNIVERSE")
        self.universe = Universe(self.problem, self.universe_output_directory, universe_seed)


    def run(individuals=None):
        output = self.universe.run(self.problem)
        # TODO do we want to save anything?
        return output


    def close():
        ezLogging.logging_remove_handler(log_handler_2file)
        del self.universe
        del self.problem
        gc.collect()