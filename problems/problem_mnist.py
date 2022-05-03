'''
Toy problem to play with NAS for MNIST
'''
### packages
import os
import glob
import numpy as np
import tensorflow as tf
import pickle as pkl
from copy import deepcopy
import pdb

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems.problem_definition import ProblemDefinition_Abstract, welless_check_decorator
from codes.factory import FactoryDefinition
from data.data_tools.loader import ezDataLoader_MNIST
from codes.utilities.custom_logging import ezLogging
from post_process import save_things, plot_things
from codes.utilities import decorators
# Block Defs
from codes.block_definitions.shapemeta.block_shapemeta import BlockShapeMeta_TFKeras
from codes.block_definitions.operators.block_operators import BlockOperators_TFKeras
from codes.block_definitions.arguments.block_arguments import BlockArguments_Auto
from codes.block_definitions.evaluate.block_evaluate_graph import BlockEvaluate_TFKeras
from codes.block_definitions.mutate.block_mutate import BlockMutate_OptB
from codes.block_definitions.mate.block_mate import BlockMate_WholeOnly
# Individual Defs
from codes.individual_definitions.individual_mutate import IndividualMutate_RollOnEachBlock
from codes.individual_definitions.individual_mate import IndividualMate_RollOnEachBlock
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_Standard



class Problem(ProblemDefinition_Abstract):
    '''
    Vanilla/basic usage of Augmentor pipeline + tf.keras for CIFAR10

    Basic block flow:
        1) Data Augmentation only for training (increase dataset size)
        2) Data Preprocessing
        3) Custom Keras NN
    '''
    def __init__(self):
        self.check_set_gpu()

        population_size = 4
        number_universe = 1
        factory = FactoryDefinition
        factory_instance = factory()
        mpi = False
        genome_seeds = []
        super().__init__(population_size, number_universe, factory, mpi, genome_seeds)

        conv_block_def = self.construct_block_def(nickname="ConvLayers",
                                                  shape_def=BlockShapeMeta_TFKeras,
                                                  operator_def=BlockOperators_TFKeras,
                                                  argument_def=BlockArguments_Auto(BlockOperators_TFKeras(), 4),
                                                  evaluate_def=BlockEvaluate_TFKeras,
                                                  mutate_def=BlockMutate_OptB(prob_mutate=0.2, num_mutants=1),
                                                  mate_def=BlockMate_WholeOnly(prob_mate=1/2))

        self.construct_individual_def(block_defs=[conv_block_def],
                                      mutate_def=IndividualMutate_RollOnEachBlock,
                                      mate_def=IndividualMate_RollOnEachBlock,
                                      evaluate_def=IndividualEvaluate_Standard)


    def check_set_gpu(self):
        '''
        https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
        '''
        import tensorflow as tf
        if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
            ezLogging.critical("GPU NOT FOUND - ezCGP EXITING")
            pdb.set_trace()

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


    @decorators.stopwatch_decorator
    def construct_dataset(self):
        loader = ezDataLoader_MNIST()
        self.training_datalist, self.validating_datalist, self.testing_datalist = loader.load()


    def set_optimization_goals(self):
        self.maximize_objectives = [True, True]
        self.objective_names = ["Accuracy", "F1"] # will be helpful for plotting later


    @decorators.stopwatch_decorator
    @welless_check_decorator
    def objective_functions(self, indiv):
        '''
        :param indiv: individual which contains references to output of training
        :return: None

        2 objectives:
            1) accuracy score
            2) f1 score

        With updated code, we expect the last block to return the validation metrics assigned to the Model object,
        so we just need to connect those to the individual's fitness values
        '''
        indiv.fitness.values = tuple(indiv.output)


    def check_convergence(self, universe):
        GENERATION_LIMIT = 1 #50

        # only going to look at the 2nd objective value which is f1
        min_firstobjective_index = universe.pop_fitness_scores[:,0].argmin()
        min_firstobjective = universe.pop_fitness_scores[min_firstobjective_index,:]
        ezLogging.warning("Checking Convergence - generation %i, best score: %s" % (universe.generation, min_firstobjective))

        if universe.generation >= GENERATION_LIMIT:
            ezLogging.warning("TERMINATING...reached generation limit.")
            universe.converged = True


    def postprocess_generation(self, universe):
        '''
        Save fitness scores and the refiners on the pareto front of fitness scroes
        '''
        ezLogging.info("Post Processing Generation Run")

        save_things.save_fitness_scores(universe)
        save_things.save_HOF_scores(universe)
        save_things.save_population(universe)


    def postprocess_universe(self, universe):
        # ezLogging.info("Post Processing Universe Run")
        # save_things.save_population(universe)
        # save_things.save_population_asLisp(universe, self.indiv_def)
        pass