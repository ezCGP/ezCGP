### packages
import os
import numpy as np
import logging
import torch
import pickle as pkl
from copy import deepcopy
import torch

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems import problem_simgan
from data.data_tools import simganData
from problems.problem_definition import ProblemDefinition_Abstract, welless_check_decorator
from codes.utilities.custom_logging import ezLogging
from codes.utilities.gan_tournament_selection import get_graph_ratings
import codes.utilities.simgan_feature_eval as feature_eval
from codes.utilities.simgan_fid_metric import get_fid_scores
from codes.utilities.simgan_support_size_eval import get_support_size
from post_process import save_things
from post_process import plot_things
from post_process import plot_signals
from codes.utilities import decorators



class Problem(problem_simgan.Problem):
    """
    Basically the same as the other simgan problem but we want to use a different dataset.
    This allows us to toggle between them really easily.
    """
    def __init__(self):
        super().__init__()


    def construct_dataset(self):
        """
        Constructs a train and validation 1D signal datasets
        """
        # Can configure the real and simulated sizes + batch size, but we will use default
        train_config_dict = {"device": "cuda",
                             "offline_mode": False}  # was gpu but that didn't work anymore
        self.training_datalist = [simganData.TransformSimGANDataset(real_size=512, sim_size=128**2, batch_size=128),
                                  train_config_dict]
        self.validating_datalist = [simganData.TransformSimGANDataset(real_size=128, sim_size=int((128**2)/4), batch_size=128)]


    def check_convergence(self, universe):
        '''
        TODO: add code for determining whether convergence has been reached
        '''
        GENERATION_LIMIT = 1 # TODO
        if universe.generation >= GENERATION_LIMIT:
            ezLogging.warning("TERMINATING...reached generation limit.")
            universe.converged = True


    def postprocess_generation(self, universe):
        '''
        Save fitness scores and the refiners on the pareto front of fitness scroes
        '''
        # run the simgan_problem.py postprocess_generation first
        super().postprocess_generation(universe)

        # now do mse stuff
        fitlist = []
        mses=[]
        for individual in universe.population.population:
            if not individual.dead:
                # print meta eval
                if hasattr(individual, 'mse'):
                    mses.append(individual.mse[-1])
                    fitlist.append(individual.fitness.values)
                    ezLogging.warning("Meta eval mse: %s fitness scores: %s" % (individual.mse, individual.fitness.values))
                else:
                    ezLogging.warning("No mse")

        mses = np.array(mses)
        fitlist = np.array(fitlist)
        for i in range(len(self.maximize_objectives)):
            plot_things.plot_mse_metric(mses, 
                                        fitlist, 
                                        objective_names=self.objective_names,
                                        maximize_objectives=self.maximize_objectives,
                                        fitness_index=i,
                                        save_path=universe.output_folder)
