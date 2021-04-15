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
from codes.block_definitions.shapemeta.block_shapemeta import BlockShapeMeta_SymbolicRegressionArg_ProbActive
from codes.block_definitions.operators.block_operators import BlockOperators_SymbRegressionOpsForArraysNoArgs
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
        population_size = 2**9
        number_universe = 1
        factory = FactoryDefinition
        mpi = True
        genome_seeds = glob.glob("outputs/problem_symbolicRegression_probactive/20210402-162139/univ0000/gen_0548_*.pkl")
        if len(genome_seeds)==0:
            pdb.set_trace()
        super().__init__(population_size, number_universe, factory, mpi, genome_seeds)

        block_def = self.construct_block_def(nickname = "main_block",
                                             shape_def = BlockShapeMeta_SymbolicRegressionArg_ProbActive,
                                             operator_def = BlockOperators_SymbRegressionOpsForArraysNoArgs,
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
        # fille node 4
        x = [4, 3, 2, 1]
        y = [47, 1580, 35136, 454464]
        nodes = ezData.ezData_numpy(x, y)
        previous_1 = ezData.ezData_numpy([1, 47, 1580, 35136])
        previous_2 = ezData.ezData_numpy([0, 1, 47, 1580])
        previous_3 = ezData.ezData_numpy([0, 0, 1, 47])
        previous_4 = ezData.ezData_numpy([0, 0, 0, 1])
        nodes_left = ezData.ezData_numpy([4+2, 5, 4, 3])
        combos = ezData.ezData_numpy([11, 9, 7, 5])
        total_genomes = ezData.ezData_numpy([180, 4500, 72000, 648000])
        previous_genome1 = ezData.ezData_numpy([5, 180, 4500, 72000])
        previous_genome2 = ezData.ezData_numpy([0, 5, 180, 4500])
        previous_genome3 = ezData.ezData_numpy([0, 0, 5, 180])
        previous_genome4 = ezData.ezData_numpy([0, 0, 0, 5])
        ephemeral_constant_1 = ezData.ezData_numpy([1]*4)
        ephemeral_constant_2 = ezData.ezData_numpy([2]*4)

        self.training_datalist = [nodes,
                                  previous_1,
                                  previous_2,
                                  previous_3,
                                  previous_4,
                                  nodes_left,
                                  combos,
                                  total_genomes,
                                  previous_genome1,
                                  previous_genome2,
                                  previous_genome3,
                                  previous_genome4,
                                  ephemeral_constant_1,
                                  ephemeral_constant_2]
        self.validating_datalist = None


    def objective_functions(self, indiv):
        if indiv.dead:
            indiv.fitness.values = (np.inf, np.inf)
        else:
            actual = self.training_datalist[0].y
            training_output, validating_output = indiv.output
            predict = training_output[0]
            error = np.log(actual)-np.log(predict)
            rms_error = np.sqrt(np.mean(np.square(error)))
            if np.isnan(rms_error):
                rms_error = np.inf
            max_error = np.max(np.abs(error))
            if np.isnan(max_error):
                max_error = np.inf
            indiv.fitness.values = (rms_error, max_error)


    def check_convergence(self, universe):
        GENERATION_LIMIT = 9000
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
        '''
        grab the individual with lowest rmse and 
        '''
        min_firstobjective_index = universe.pop_fitness_scores[:,0].argmin()
        best_rmse = universe.pop_fitness_scores[min_firstobjective_index,0]
        best_indiv_id = universe.pop_individual_ids[min_firstobjective_index]
        found = False
        for indiv in universe.population.population:
            if indiv.id == best_indiv_id:
                found = True
                break

        actual = self.training_datalist[0].y
        predict = indiv.output[0][0]
        x = self.training_datalist[0].x
        plt.figure(figsize=(15,10))
        plt.plot(x, actual, marker='x', label="actual")
        plt.plot(x, predict, marker='x', label="predict")
        plt.legend()
        plt.title("Generation %i - RMSE %.2f" % (universe.generation, best_rmse))
        plt.savefig(os.path.join(universe.output_folder, "best_indiv_gen%04d" % universe.generation))
        plt.close()

        ezLogging.warning("PostProcess Gen %i - %s" % (universe.generation, predict))
        save_things.save_fitness_scores(universe)
        save_things.save_population(universe)


    def postprocess_universe(self, universe):
        pdb.set_trace()