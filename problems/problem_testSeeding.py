'''
root/problems/problem_multiGaussian.py
'''

### packages
import os
import numpy as np
import glob
import logging

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems.problem_definition import ProblemDefinition_Abstract
from codes.factory import FactoryDefinition
from data.data_tools import ezData
from codes.block_definitions.block_shapemeta import BlockShapeMeta_Gaussian
from codes.block_definitions.block_operators import BlockOperators_Gaussian
from codes.block_definitions.block_arguments import BlockArguments_Gaussian
from codes.block_definitions.block_evaluate import BlockEvaluate_Standard
from codes.block_definitions.block_mutate import BlockMutate_NoFtn
from codes.block_definitions.block_mate import BlockMate_NoMate
from codes.individual_definitions.individual_mutate import IndividualMutate_RollOnEachBlock
from codes.individual_definitions.individual_mate import IndividualMate_RollOnEachBlock
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_Standard
from post_process import save_things
from post_process import plot_things
from codes.utilities.custom_logging import ezLogging



class Problem(ProblemDefinition_Abstract):
    '''
    Not intented to see if this does a good job at evolving but rather just a quick way to test out the different
    mating, mutating, operators etc with multiple blocks.
    '''
    def __init__(self):
        population_size = 52 #must be divisible by 4 if doing mating
        number_universe = 1 #10
        factory = FactoryDefinition
        mpi = False
        genome_seeds = glob.glob(os.path.join(os.getcwd(),
                                              "outputs/problem_testSeeding/20201115-171924-generate_seeds/univ0000/gen_0003_*.pkl"))
        super().__init__(population_size, number_universe, factory, mpi, genome_seeds)

        block_def = self.construct_block_def(nickname = "GaussBlock",
                                             shape_def = BlockShapeMeta_Gaussian, #maybe have x2 num of gaussians so 20
                                             operator_def = BlockOperators_Gaussian, #only 1 operator...gauss taking in th right args
                                             argument_def = BlockArguments_Gaussian, #0-100 floats, 0-1 floats, 0-100 ints
                                             evaluate_def = BlockEvaluate_Standard, #ya standard eval
                                             mutate_def = BlockMutate_NoFtn, #maybe not mutate ftn
                                             mate_def = BlockMate_NoMate) #maybe not mate

        self.construct_individual_def(block_defs = [block_def],
                                      mutate_def = IndividualMutate_RollOnEachBlock,
                                      mate_def = IndividualMate_RollOnEachBlock,
                                      evaluate_def = IndividualEvaluate_Standard)
        # where to put this?
        self.construct_dataset()


    def construct_dataset(self):
        from misc import fake_mixturegauss
        x, y, noisy, goal_features = fake_mixturegauss.main()
        x = fake_mixturegauss.XLocations(x)
        starting_sum = fake_mixturegauss.RollingSum(np.zeros(x.shape))
        #self.data = data_loader.load_symbolicRegression([x, starting_sum], [y, noisy, goal_features])
        self.train_data = ezData.ezData([x, starting_sum], [y, noisy, goal_features])
        self.validate_data = None


    def objective_functions(self, indiv):
        if indiv.dead:
            indiv.fitness.values = (np.inf, np.inf, np.inf)
        else:
            clean_y, noisy_y, goal_features = self.train_data.y
            predict_y = indiv.output[0]
            # how to extract the arguments to match to goal_features as well?
            error = clean_y-predict_y
            rms_error = np.sqrt(np.mean(np.square(error)))
            max_error = np.max(np.abs(error))
            # YO active nodes includes outputs and input nodes so 10 main nodes + 2 inputs + 1 output
            #active_error = np.abs(10+2+1-len(indiv[0].active_nodes)) #maybe cheating by knowing the goal amount ahead of time
            active_error = len(indiv[0].active_nodes)
            indiv.fitness.values = (rms_error, max_error, active_error)


    def check_convergence(self, universe):
        GENERATION_LIMIT = 3 #1000
        SCORE_MIN = 1e-1

        # only going to look at the first objective value which is rmse
        # CAREFUL, after we added the ids, the values are now strings not floats
        min_firstobjective_index = universe.pop_fitness_scores[:,0].astype(float).argmin()
        min_firstobjective = universe.pop_fitness_scores[min_firstobjective_index,:-1].astype(float)
        logging.warning("Checking Convergence - generation %i, best score: %s" % (universe.generation, min_firstobjective))

        if universe.generation >= GENERATION_LIMIT:
            logging.warning("TERMINATING...reached generation limit.")
            universe.converged = True
        if min_firstobjective[0] < SCORE_MIN:
            logging.warning("TERMINATING...reached minimum scores.")
            universe.converged = True


    def postprocess_generation(self, universe):
        '''
        save scores and population
        '''
        logging.info("Post Processing Generation Run")
        save_things.save_fitness_scores(universe)
        save_things.save_population(universe)

