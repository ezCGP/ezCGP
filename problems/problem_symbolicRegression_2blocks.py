'''
root/problems/problem_symbolicRegression.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
import numpy as np
#import logging

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from codes.utilities.custom_logging import ezLogging
from problems.problem_definition import ProblemDefinition_Abstract
from codes.factory import FactoryDefinition
from data.data_tools import ezData
from codes.block_definitions.shapemeta.block_shapemeta import BlockShapeMeta_SymbolicRegressionNoArg25, BlockShapeMeta_SymbolicRegressionArg25
from codes.block_definitions.operators.block_operators import BlockOperators_SymbRegressionOpsNoArgs, BlockOperators_SymbRegressionOpsWithArgs
from codes.block_definitions.arguments.block_arguments import BlockArguments_NoArgs, BlockArguments_SmallFloatOnly
from codes.block_definitions.evaluate.block_evaluate import BlockEvaluate_MiddleBlock, BlockEvaluate_FinalBlock
from codes.block_definitions.mutate.block_mutate import BlockMutate_OptA, BlockMutate_OptB
from codes.block_definitions.mate.block_mate import BlockMate_WholeOnly
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
                                             argument_def = BlockArguments_NoArgs,
                                             evaluate_def = BlockEvaluate_MiddleBlock,
                                             mutate_def = BlockMutate_OptA,
                                             mate_def = BlockMate_WholeOnly)

        block1_def = self.construct_block_def(nickname = "wArg_block",
                                             shape_def = BlockShapeMeta_SymbolicRegressionArg25,
                                             operator_def = BlockOperators_SymbRegressionOpsWithArgs,
                                             argument_def = BlockArguments_SmallFloatOnly,
                                             evaluate_def = BlockEvaluate_FinalBlock,
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
        x = np.random.uniform(low=0.25, high=2, size=200)
        y = self.goal_function(x)
        data = ezData.ezData_numpy(x, y)
        ephemeral_constant = ezData.ezData_float(1)

        self.training_datalist = [ephemeral_constant, data]
        self.validating_datalist = None


    def objective_functions(self, indiv):
        if indiv.dead:
            indiv.fitness.values = (np.inf, np.inf)
        else:
            actual = self.training_datalist[-1].y
            training_output, validating_output = indiv.output
            predict = training_output[0]
            error = actual-predict
            rms_error = np.sqrt(np.mean(np.square(error)))
            max_error = np.max(np.abs(error))
            indiv.fitness.values = (rms_error, max_error)


    def check_convergence(self, universe):
        GENERATION_LIMIT = 5
        SCORE_MIN = 1e-1

        # only going to look at the first objective value which is rmse
        min_firstobjective_index = universe.pop_fitness_scores[:,0].argmin()
        min_firstobjective = universe.pop_fitness_scores[min_firstobjective_index,:-1]
        ezLogging.warning("Checking Convergence - generation %i, best score: %s" % (universe.generation, min_firstobjective))

        if universe.generation >= GENERATION_LIMIT:
            ezLogging.warning("TERMINATING...reached generation limit.")
            universe.converged = True
        if min_firstobjective[0] < SCORE_MIN:
            ezLogging.warning("TERMINATING...reached minimum scores.")
            universe.converged = True
