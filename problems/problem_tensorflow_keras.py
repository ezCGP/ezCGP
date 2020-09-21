
import numpy as np
import logging

from problems.problem_definition import ProblemDefinition_Abstract

from codes.factory import FactoryDefinition
from codes.block_definitions.block_shapemeta import BlockShapeMeta_Keras

from codes.individual_definitions.individual_mutate import IndividualMutate_RollOnEachBlock
from codes.individual_definitions.individual_mate import IndividualMate_RollOnEachBlock
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_Standard

from data.data_tools.data_loader import load_CIFAR10

# Fitness imports
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score as accuracy

class Problem(ProblemDefinition_Abstract):
    def __init__(self):
        population_size = 8
        number_universe = 1
        factory = FactoryDefinition
        factory_instance = factory()
        mpi = True
        super().__init__(population_size, number_universe, factory, mpi)

        block_def = self.construct_block_def(nickname = "main_block",
                                             shape_def = BlockShapeMeta_Keras,
                                             operator_def =  TFOps,
                                             argument_def = TFArgs,
                                             evaluate_def = BlockTensorFlowEvaluate,
                                             mutate_def = BlockMutateA,
                                             mate_def = BlockNoMate)

        self.construct_individual_def(block_defs = [block_def],
                                    mutate_def=IndividualMutate_RollOnEachBlock,
                                    mate_def=IndividualMate_RollOnEachBlock,
                                    evaluate_def=IndividualEvaluate_Standard)

        # where to put this?
        self.construct_dataset()


    def goal_function(self, data):
        # TODO what is this
        return 1/data

    def construct_dataset(self):
        """
        Loads cifar 10
        :return: None
        """
        dataset = load_CIFAR10(.8, .2)

        # force normalization  # will now apply to both pipelines
        # dataset.preprocess_pipeline.add_operation(Normalize())

        self.data = dataset

    def objective_functions(self, indiv):
        """
        :param indiv: individual which contains references to output of training
        :return: None
        """
        dataset = self.data
        _, actual = dataset.preprocess_test_data()
        actual = np.argmax(actual, axis = 1)
        predict = indiv.output
        predict = np.argmax(predict, axis = 1)
        acc_score = accuracy(actual, predict)
        f1 = f1_score(actual, predict, average = "macro")
        indiv.fitness.values = (-acc_score, -f1)  # want to minimize this

    def check_convergence(self, universe):
        """
        :param universe:
        :return:
        """
        GENERATION_LIMIT = 1
        SCORE_MIN = 1e-1

        print("\n\n\n\n\n", universe.generation, np.min(np.array(universe.fitness_scores)))

        if universe.generation >= GENERATION_LIMIT:
            print("TERMINATING...reached generation limit")
            universe.converged = True
        if np.min(np.array(universe.fitness_scores)[0]) < SCORE_MIN:
            print("TERMINATING...reached minimum score")
            universe.converged = True