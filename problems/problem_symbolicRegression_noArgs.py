'''
root/problems/problem_symbolicRegression.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
import numpy as np

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problem.problem_abstract import ProblemDefinition
from codes.factory import Factory
from data.data_tools import data_loader
from codes.block_definitions.block_shapemeta import BlockShapeMeta_SymbolicRegression25
from codes.block_definitions.block_operators import BlockOperators_SymbRegressionOpsNoArgs
from codes.block_definitions.block_arguments import BlockArgumentsNoArgs
from codes.block_definitions.block_evaluate import BlockEvaluate_Standard
from codes.block_definitions.block_mutate import BlockMutate_OptA
from codes.block_definitions.block_mate import BlockMate_NoMate
from codes.individual_definitions.individual_mutate import IndividualMutate_RollOnEachBlock
from codes.individual_definitions.individual_mate import IndividualMate_RollOnEachBlock
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_Standard



class Problem(ProblemDefinition):
    '''
    TODO
    '''
    def __init__(self):
        population_size = 8
        number_universe = 1
        factory = Factory
        factory_instance = factory()
        mpi = False
        super().__init__(population_size, number_universe, factory, mpi)

        block_def = self.construct_block_def(nickname = "main_block",
                                             shape_def = BlockShapeMeta_SymbolicRegression25,
                                             operator_def = BlockOperators_SymbRegressionOpsNoArgs,
                                             argument_def = BlockArgumentsNoArgs,
                                             evaluate_def = BlockEvaluate_Standard,
                                             mutate_def = BlockMutate_OptA,
                                             mate_def = BlockMate_NoMate)

        self.construct_individual_def(block_defs = [block_def],
                                      mutate_def = IndividualMutate_RollOnEachBlock,
                                      mate_def = IndividualMate_RollOnEachBlock,
                                      evaluate_def = IndividualEvaluate_Standard)

        # where to put this?
        self.construct_dataset()


    def goal_function(self, data):
        return 1/data


    def construct_dataset(self):
        x = [np.float64(1), np.random.uniform(low=0.25, high=2, size=200)]
        y = self.goal_function(self.x_train[1])
        self.data = data_loader.load_symbolicRegression(x, y)


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