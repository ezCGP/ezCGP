'''
root/problems/problem_simgan_ecg.py
'''

### packages
import os
import numpy as np
import logging

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems.problem_definition import ProblemDefinition_Abstract
from problems import problem_simgan
from codes.utilities import decorators
from data.data_tools import simganData
from codes.utilities.custom_logging import ezLogging
from codes.factory import Factory_SimGAN_ECG
from codes.block_definitions.shapemeta.block_shapemeta import BlockShapeMeta_SimGAN_Network, BlockShapeMeta_SimGAN_Train_Config
from codes.block_definitions.operators.block_operators import BlockOperators_SimGAN_Refiner, BlockOperators_SimGAN_Discriminator, BlockOperators_SimGAN_ECG_Train_Config
from codes.block_definitions.arguments.block_arguments import BlockArguments_Auto
from codes.block_definitions.evaluate.block_evaluate_pytorch import BlockEvaluate_SimGAN_Refiner, BlockEvaluate_SimGAN_Discriminator, BlockEvaluate_SimGAN_Train_Config
from codes.block_definitions.mutate.block_mutate import BlockMutate_OptB_No_Single_Ftn, BlockMutate_OptB, BlockMutate_ArgsOnly
from codes.block_definitions.mate.block_mate import BlockMate_WholeOnly
from codes.individual_definitions.individual_mutate import IndividualMutate_RollOnEachBlock_LimitedMutants
from codes.individual_definitions.individual_mate import IndividualMate_RollOnEachBlock
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_SimGAN



class Problem(problem_simgan.Problem):
    def __init__(self):
        '''
        not gonna init problem_simgan.Problem since we want our genome and block defs to be a little different
        '''
        population_size = 4 # TODO #must be divisible by 4 if doing mating
        number_universe = 1
        factory = Factory_SimGAN_ECG
        mpi = False
        genome_seeds = [["misc/IndivSeed_SimGAN_Seed0/RefinerBlock_lisp.txt",
                         "misc/IndivSeed_SimGAN_Seed0/DiscriminatorBlock_lisp.txt",
                         "misc/IndivSeed_SimGAN_ECG_Seed0/ConfigBlock_lisp.txt"]]*population_size
        hall_of_fame_flag = True
        super(problem_simgan.Problem, self).__init__(population_size, number_universe, factory, mpi, genome_seeds, hall_of_fame_flag)
        self.relativeScoring = True # this will force universe to be instance of RelativePopulationUniverseDefinition() in main.py

        refiner_def = self.construct_block_def(nickname = "refiner_block",
                                               shape_def = BlockShapeMeta_SimGAN_Network,
                                               operator_def = BlockOperators_SimGAN_Refiner,
                                               argument_def = BlockArguments_Auto(BlockOperators_SimGAN_Refiner(), 10),
                                               evaluate_def = BlockEvaluate_SimGAN_Refiner,
                                               mutate_def=BlockMutate_OptB_No_Single_Ftn(prob_mutate=0.2, num_mutants=2),
                                               mate_def=BlockMate_WholeOnly(prob_mate=1/3)
                                              )

        discriminator_def = self.construct_block_def(nickname = "discriminator_block",
                                                     shape_def = BlockShapeMeta_SimGAN_Network,
                                                     operator_def = BlockOperators_SimGAN_Discriminator,
                                                     argument_def = BlockArguments_Auto(BlockOperators_SimGAN_Discriminator(), 15),
                                                     evaluate_def = BlockEvaluate_SimGAN_Discriminator,
                                                     mutate_def=BlockMutate_OptB(prob_mutate=0.2, num_mutants=2),
                                                     mate_def=BlockMate_WholeOnly(prob_mate=1/3)
                                                    )

        train_config_def = self.construct_block_def(nickname = "train_config",
                                                    shape_def = BlockShapeMeta_SimGAN_Train_Config,
                                                    operator_def = BlockOperators_SimGAN_ECG_Train_Config,
                                                    argument_def = BlockArguments_Auto(BlockOperators_SimGAN_ECG_Train_Config(), 10),
                                                    evaluate_def = BlockEvaluate_SimGAN_Train_Config,
                                                    mutate_def=BlockMutate_ArgsOnly(prob_mutate=0.1, num_mutants=2),
                                                    mate_def=BlockMate_WholeOnly(prob_mate=1/3)
                                                   )

        self.construct_individual_def(block_defs = [refiner_def, discriminator_def, train_config_def],
                                      mutate_def = IndividualMutate_RollOnEachBlock_LimitedMutants,
                                      mate_def = IndividualMate_RollOnEachBlock,
                                      evaluate_def = IndividualEvaluate_SimGAN
                                      )
        self.construct_dataset()


    @decorators.stopwatch_decorator
    def construct_dataset(self):
        '''
        Constructs a train and validation 1D signal datasets
        '''
        # Can configure the real and simulated sizes + batch size, but we will use default
        train_config_dict = {'device': 'cuda',
                             'offline_mode': False}
        self.training_datalist = [simganData.SimGANECGDataset(real_size=512, sim_size=128**2, batch_size=4),
                                  train_config_dict]
        self.validating_datalist = [simganData.SimGANECGDataset(real_size=128, sim_size=int((128**2)/4), batch_size=4)]


    def check_convergence(self, universe):
        '''
        TODO: add code for determining whether convergence has been reached
        '''
        GENERATION_LIMIT = 1 # TODO
        if universe.generation >= GENERATION_LIMIT:
            ezLogging.warning("TERMINATING...reached generation limit.")
            universe.converged = True
