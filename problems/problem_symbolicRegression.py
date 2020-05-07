# packages
import numpy as np

# scripts
from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__)))) # ezCGP/problems/__file__
from problem.problem_abstract import ProblemDefinition
from code.factory import Factory
from operators import SymbRegressionNoArgs
from arguments import NoArgs
from evaluate import IndividualStandardEvaluate, BlockStandardEvaluate
from mutate import InidividualMutateA, BlockMutateA
from mate import IndividualMateA, BlockNoMate
from ezData.data_pair import DataPair
from database.ezDataLoader import load_symbolicRegression

class Problem(ProblemDefinition):
    def __init__(self):
        population_size = 8
        number_universe = 1
        factory = Factory
        factory_instance = factory()
        mpi = False
        super().__init__(population_size, number_universe, factory, mpi)

        block_def = self.construct_block_def(nickname = "main_block",
                                            shape_def = ShapeA,
                                            operator_def = SymbRegressionNoArgs,
                                            argument_def = NoArgs,
                                            evaluate_def = BlockStandardEvaluate,
                                            mutate_def = BlockMutateA,
                                            mate_def = BlockNoMate)

        self.construct_individual_def(block_defs = [block_def],
                                    mutate_def = InidividualMutateA,
                                    mate_def = IndividualMateA,
                                    evaluate_def = IndividualStandardEvaluate)

        # where to put this?
        self.construct_dataset()


    def goal_function(self, data):
        return 1/data

    def construct_dataset(self):
        x = [np.float64(1), np.random.uniform(low=0.25, high=2, size=200)]
        y = self.goal_function(self.x_train[1])
        self.data = load_symbolicRegression(x, y)

    def objective_functions(self, indiv):
        actual = self.data.y_train

        predit = indiv.output
        error = actual-predit
        rms_error = np.sqrt(np.mean(np.square(error)))
        max_error = np.max(np.abs(error))
        indiv.fitness.values = (rms_error, max_error)

    def check_convergence(self, universe):
        GENERATION_LIMIT = 10
        SCORE_MIN = 1e-1

        print("\n\n\n\n\n", universe.generation, np.min(np.array(universe.fitness_scores)))

        if universe.generation >= GENERATION_LIMIT:
            print("TERMINATING...reached generation limit")
            universe.converged = True
        if np.min(np.array(universe.fitness_scores)[0]) < SCORE_MIN:
            print("TERMINATING...reached minimum score")
            universe.converged = True