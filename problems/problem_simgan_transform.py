### packages
import os
import numpy as np
import logging
import torch
import pickle as pkl
from copy import deepcopy

### sys relative to root dir
import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems import problem_simgan

from problems.problem_definition import (
    ProblemDefinition_Abstract,
    welless_check_decorator,
)
from codes.factory import Factory_SimGAN
from data.data_tools import simganData


class Problem(problem_simgan.Problem):
    """
    Not intented to see if this does a good job at evolving but rather just a quick way to test out the different
    mating, mutating, operators etc with multiple blocks.
    """

    def construct_dataset(self):
        """
        Constructs a train and validation 1D signal datasets
        """
        # Can configure the real and simulated sizes + batch size, but we will use default
        train_config_dict = {"device": "cuda"}  # was gpu but that didn't work anymore
        self.training_datalist = [
            simganData.TransformSimGANDataset(
                real_size=512, sim_size=128**2, batch_size=128
            ),
            train_config_dict,
        ]
        self.validating_datalist = [
            simganData.TransformSimGANDataset(
                real_size=128, sim_size=int((128**2) / 4), batch_size=128
            )
        ]
    def set_optimization_goals(self):
        self.maximize_objectives = [False, False, False, False, False, True, True]

    @welless_check_decorator
    def objective_functions(self, population):
        """
        Get the best refiner and discriminator from each individual in the population and do a tournament selection to rate them
        # TODO: add in the support size as a metric
        """
        n_individuals = len(population.population)
        refiners = []
        discriminators = []
        alive_individual_index = []
        for i, indiv in enumerate(population.population):
            if not indiv.dead:
                alive_individual_index.append(i)
                R, D = indiv.output
                print(indiv.output)
                refiners.append(R.cpu())
                discriminators.append(D.cpu())

        # Run tournament and add ratings
        if len(alive_individual_index) > 0:
            #  Objective #1 - NO LONGER AN OBJECTIVE FOR POPULATION SELECTION
            refiner_ratings, _ = get_graph_ratings(
                refiners, discriminators, self.validating_datalist[0], "cpu"
            )
            #  Objective #2
            refiner_fids = get_fid_scores(refiners, self.validating_datalist[0])

            # Objective #3, #4, #5
            refiner_feature_dist = feature_eval.calc_feature_distances(
                refiners, self.validating_datalist[0], "cpu"
            )

            # Objective #6, #7
            refiner_t_tests = feature_eval.calc_t_tests(
                refiners, self.validating_datalist[0], "cpu"
            )

            # Objective #8
            support_size = get_support_size(
                refiners, self.validating_datalist[0], "cpu"
            )

            for (
                indx,
                rating,
                fid,


