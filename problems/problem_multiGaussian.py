'''
root/problems/problem_multiGaussian.py
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
from codes.block_definitions.block_shapemeta import BlockShapeMeta_Gaussian
from codes.block_definitions.block_operators import 
from codes.block_definitions.block_arguments import BlockArguments_Gaussian
from codes.block_definitions.block_evaluate import BlockEvaluate_Standard
from codes.block_definitions.block_mutate import 
from codes.block_definitions.block_mate import
from codes.individual_definitions.individual_mutate import IndividualMutate_RollOnEachBlock
from codes.individual_definitions.individual_mate import IndividualMate_RollOnEachBlock
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_Standard
from post_process import save_things



class Problem(ProblemDefinition_Abstract):
    '''
    Not intented to see if this does a good job at evolving but rather just a quick way to test out the different
    mating, mutating, operators etc with multiple blocks.
    '''
    def __init__(self):
        population_size = 12 #must be divisible by 4 if doing mating
        number_universe = 1
        factory = FactoryDefinition
        mpi = False
        super().__init__(population_size, number_universe, factory, mpi)

        block_def = self.construct_block_def(nickname = "wArg_block",
                                             shape_def = BlockShapeMeta_Gaussian, #maybe have x2 num of gaussians so 20
                                             operator_def = BlockOperators_SymbRegressionOpsWithArgs, #only 1 operator...gauss taking in th right args
                                             argument_def = BlockArguments_Gaussian, #0-100 floats, 0-1 floats, 0-100 ints
                                             evaluate_def = BlockEvaluate_Standard, #ya standard eval
                                             mutate_def = BlockMutate_OptB, #maybe not mutate ftn
                                             mate_def = BlockMate_WholeOnly) #maybe not mate

        self.construct_individual_def(block_defs = [block_def],
                                      mutate_def = IndividualMutate_RollOnEachBlock,
                                      mate_def = IndividualMate_RollOnEachBlock,
                                      evaluate_def = IndividualEvaluate_Standard)

        # where to put this?
        self.construct_dataset()


    def goal_function(self, data):
        return 1/data


    def construct_dataset(self):
        from misc import fake_mixturegauss, RollingSum, XLocations
        x, y, noisy, goal_features = fake_mixturegauss.main()
        x = XLocations(x)
        starting_sum = RollingSum(np.zeros(x.shape))
        self.data = data_loader.load_symbolicRegression([x, starting_sum], [y, noisy, goal_features])


    def objective_functions(self, indiv):
        if indiv.dead:
            indiv.fitness.values = (np.inf, np.inf)
        else:
            clean_y, noisy_y, goal_features = self.data.y_train
            predict_y = indiv.output
            # how to extract the arguments to match to goal_features as well?
            error = clean_y-predict_y
            rms_error = np.sqrt(np.mean(np.square(error)))
            max_error = np.max(np.abs(error))
            indiv.fitness.values = (rms_error, max_error)


    def check_convergence(self, universe):
        GENERATION_LIMIT = 2
        SCORE_MIN = 1e-1

        # only going to look at the first objective value which is rmse
        min_firstobjective_index = universe.fitness_scores[:,0].argmin()
        min_firstobjective = universe.fitness_scores[min_firstobjective_index,:-1].astype(float)
        logging.warning("Checking Convergence - generation %i, best score: %s" % (universe.generation, min_firstobjective))

        if universe.generation >= GENERATION_LIMIT:
            logging.warning("TERMINATING...reached generation limit.")
            universe.converged = True
        if min_firstobjective[0] < SCORE_MIN:
            logging.warning("TERMINATING...reached minimum scores.")
            universe.converged = True


    def postprocess_generation(self, universe):
        '''
        I'd say just store an archive of scores
        '''
        logging.info("Post Processing Generation Run")
        save_things.save_fitness_scores(universe)



    def postprocess_universe(self, universe):
        '''
        save each individual at the end of the population
        '''
        logging.info("Post Processing Universe Run")
        save_things.save_population(universe)