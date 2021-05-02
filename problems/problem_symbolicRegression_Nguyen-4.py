'''
root/problems/problem_symbolicRegression.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pdb

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems.problem_definition import ProblemDefinition_Abstract
from codes.factory import FactoryDefinition
from data.data_tools import ezData
from post_process import save_things
from codes.utilities.custom_logging import ezLogging
from codes.block_definitions.shapemeta.block_shapemeta import BlockShapeMeta_SymbolicRegression_Benchmarking
from codes.block_definitions.operators.block_operators import BlockOperators_SymbolicRegression_Benchmarking
from codes.block_definitions.arguments.block_arguments import BlockArguments_NoArgs
from codes.block_definitions.evaluate.block_evaluate import BlockEvaluate_FinalBlock
from codes.block_definitions.mutate.block_mutate import BlockMutate_OptA
from codes.block_definitions.mate.block_mate import BlockMate_NoMate
from codes.individual_definitions.individual_mutate import IndividualMutate_RollOnEachBlock
from codes.individual_definitions.individual_mate import IndividualMate_RollOnEachBlock
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_Standard



class Problem(ProblemDefinition_Abstract):
    '''
    TODO
    '''
    def __init__(self):
        population_size = 2**5
        number_universe = 1
        factory = FactoryDefinition
        mpi = False
        genome_seeds = []
        super().__init__(population_size, number_universe, factory, mpi, genome_seeds)

        block_def = self.construct_block_def(nickname = "main_block",
                                             shape_def = BlockShapeMeta_SymbolicRegression_Benchmarking,
                                             operator_def = BlockOperators_SymbolicRegression_Benchmarking,
                                             argument_def = BlockArguments_NoArgs,
                                             evaluate_def = BlockEvaluate_FinalBlock,
                                             mutate_def = BlockMutate_OptA,
                                             mate_def = BlockMate_NoMate)

        self.construct_individual_def(block_defs = [block_def],
                                      mutate_def = IndividualMutate_RollOnEachBlock,
                                      mate_def = IndividualMate_RollOnEachBlock,
                                      evaluate_def = IndividualEvaluate_Standard)
        self.construct_dataset()


    def construct_dataset(self):
        '''
        check the paper for how to sample x values, and then to reconstruct the objective function
        '''
        # TODO:
        def objective_function(data):
            output = x**6 + x**5 + x**4 + x**3 + x**2 + x
            return output

        # TODO:
        x = np.random.uniform(-1, 1, 20)
        y = objective_function(x)
        dataset = ezData.ezData_numpy(x, y)
        ephemeral_constant = ezData.ezData_float(1)

        self.training_datalist = [dataset,
                                  ephemeral_constant]
        self.validating_datalist = None


    def objective_functions(self, indiv):
        '''
        The paper mentions only one objective for symbolic regression:
            "The fitness of the individuals was represented by a cost function value,
             defined as the sum of the absolute differences between the correct function
             values and the values of an evaluated individual."

        We want to minimize this cost/error
        '''
        indiv.fitness.values = (np.inf,) # default worse-possible fitness
        if not indiv.dead:
            actual = self.training_datalist[0].y
            training_output, validating_output = indiv.output
            predictted = training_output[0]

    
            if np.any(np.isnan(predicted)):
                # might as well make the individual dead and leave fitness at inf
                indiv.dead = True
            else:
                # TODO evaluate cost/error given 'actual' and 'predicted'
                error = np.sum(np.abs(np.subtract(actual, predicted)))
                indiv.fitness.values = (error,)


    def check_convergence(self, universe):
        GENERATION_LIMIT = 50
        SCORE_MIN = 0

        # only going to look at the first objective value which is rmse
        min_firstobjective_index = universe.pop_fitness_scores[:,0].argmin()
        min_firstobjective = universe.pop_fitness_scores[min_firstobjective_index,0]
        ezLogging.warning("Checking Convergence - generation %i, best score: %s" % (universe.generation, min_firstobjective))

        if universe.generation >= GENERATION_LIMIT:
            ezLogging.warning("TERMINATING...reached generation limit.")
            universe.converged = True
        if min_firstobjective < SCORE_MIN:
            ezLogging.warning("TERMINATING...reached minimum scores.")
            universe.converged = True


    def postprocess_generation(self, universe):
        ezLogging.warning("PostProcess Gen %i" % (universe.generation))
        save_things.save_fitness_scores(universe)
        save_things.save_population(universe)


    def postprocess_universe(self, universe):
        pass