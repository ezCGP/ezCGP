'''
root/problems/problem_symbolicRegression.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
import numpy as np
import logging

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems.problem_definition import ProblemDefinition_Abstract
from codes.factory import FactoryDefinition
from data.data_tools import data_loader
from codes.block_definitions.block_shapemeta import BlockShapeMeta_SymbolicRegressionNoArg25, BlockShapeMeta_SymbolicRegressionArg25
from codes.block_definitions.block_operators import BlockOperators_SymbRegressionOpsNoArgs, BlockOperators_SymbRegressionOpsWithArgs
from codes.block_definitions.block_arguments import BlockArgumentsNoArgs, BlockArgumentsSmallFloatOnly
from codes.block_definitions.block_evaluate import BlockEvaluate_Standard
from codes.block_definitions.block_mutate import BlockMutate_OptA, BlockMutate_OptB
from codes.block_definitions.block_mate import BlockMate_WholeOnly
from codes.individual_definitions.individual_mutate import IndividualMutate_RollOnEachBlock
from codes.individual_definitions.individual_mate import IndividualMate_RollOnEachBlock
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_Standard



class Problem(ProblemDefinition_Abstract):
    '''
    Not intented to see if this does a good job at evolving but rather just a quick way to test out the different
    mating, mutating, operators etc with multiple blocks.
    '''
    def __init__(self):
        population_size = 8
        number_universe = 10
        factory = FactoryDefinition
        mpi = False
        super().__init__(population_size, number_universe, factory, mpi)

        block0_def = self.construct_block_def(nickname = "wOutArg_block",
                                             shape_def = BlockShapeMeta_SymbolicRegressionNoArg25,
                                             operator_def = BlockOperators_SymbRegressionOpsNoArgs,
                                             argument_def = BlockArgumentsNoArgs,
                                             evaluate_def = BlockEvaluate_Standard,
                                             mutate_def = BlockMutate_OptA,
                                             mate_def = BlockMate_WholeOnly)

        block1_def = self.construct_block_def(nickname = "wArg_block",
                                             shape_def = BlockShapeMeta_SymbolicRegressionArg25,
                                             operator_def = BlockOperators_SymbRegressionOpsWithArgs,
                                             argument_def = BlockArgumentsSmallFloatOnly,
                                             evaluate_def = BlockEvaluate_Standard,
                                             mutate_def = BlockMutate_OptB,
                                             mate_def = BlockMate_WholeOnly)

        self.construct_individual_def(block_defs = [block0_def, block1_def],
                                      mutate_def = IndividualMutate_RollOnEachBlock,
                                      mate_def = IndividualMate_RollOnEachBlock,
                                      evaluate_def = IndividualEvaluate_Standard)

        # where to put this?
        self.construct_dataset()


    def goal_function(self, data):
        return 1/data


    def construct_dataset(self):
        x = [np.float64(1), np.random.uniform(low=0.25, high=2, size=200)]
        y = self.goal_function(x[1])
        self.data = data_loader.load_symbolicRegression(x, y)


    def objective_functions(self, indiv):
        if indiv.dead:
            indiv.fitness.values = (np.inf, np.inf)
        else:
            actual = self.data.y_train
            predict = indiv.output; print(predict)
            error = actual-predict
            rms_error = np.sqrt(np.mean(np.square(error)))
            max_error = np.max(np.abs(error))
            indiv.fitness.values = (rms_error, max_error)


    def check_convergence(self, universe):
        GENERATION_LIMIT = 100
        SCORE_MIN = 1e-1

        # only going to look at the first objective value which is rmse
        min_firstobjective_index = universe.fitness_scores[:,0].argmin()
        min_firstobjective = universe.fitness_scores[min_firstobjective_index,:-1]
        logging.warning("Checking Convergence - generation %i, best score: %s" % (universe.generation, min_firstobjective))

        if universe.generation >= GENERATION_LIMIT:
            logging.warning("TERMINATING...reached generation limit.")
            universe.converged = True
        if min_firstobjective[0] < SCORE_MIN:
            logging.warning("TERMINATING...reached minimum scores.")
            universe.converged = True