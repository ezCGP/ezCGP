### packages
import os
import numpy as np
import logging

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems.problem_definition import ProblemDefinition_Abstract
from codes.factory import FactoryDefinition
from data.data_tools import simganData
from codes.utilities.custom_logging import ezLogging
from codes.utilities.gan_tournament_selection import get_graph_ratings
from codes.block_definitions.shapemeta.block_shapemeta import BlockShapeMeta_SimGAN_Network, BlockShapeMeta_SimGAN_Train_Config
from codes.block_definitions.operators.block_operators import BlockOperators_SimGAN_Refiner, BlockOperators_SimGAN_Discriminator, BlockOperators_SimGAN_Train_Config
from codes.block_definitions.arguments.block_arguments import BlockArguments_SimGAN_Refiner, BlockArguments_SimGAN_Discriminator, BlockArguments_SimGAN_Train_Config
from codes.block_definitions.evaluate.block_evaluate_gan import BlockEvaluate_SimGAN_Refiner, BlockEvaluate_SimGAN_Discriminator, BlockEvaluate_SimGAN_Train_Config 
from codes.block_definitions.mutate.block_mutate import BlockMutate_SimGAN
from codes.block_definitions.mate.block_mate import BlockMate_WholeOnly_4Blocks
from codes.individual_definitions.individual_mutate import IndividualMutate_RollOnEachBlock
from codes.individual_definitions.individual_mate import IndividualMate_RollOnEachBlock
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_SimGAN
# from post_process import save_things
# from post_process import plot_things


class Problem(ProblemDefinition_Abstract):
    '''
    Not intented to see if this does a good job at evolving but rather just a quick way to test out the different
    mating, mutating, operators etc with multiple blocks.
    '''
    def __init__(self):
        population_size = 4 #must be divisible by 4 if doing mating
        number_universe = 1 #10
        factory = FactoryDefinition
        mpi = False
        super().__init__(population_size, number_universe, factory, mpi)
        self.isGAN = True

        refiner_def = self.construct_block_def(nickname = "refiner_block",
                                             shape_def = BlockShapeMeta_SimGAN_Network, 
                                             operator_def = BlockOperators_SimGAN_Refiner, 
                                             argument_def = BlockArguments_SimGAN_Refiner,
                                             evaluate_def = BlockEvaluate_SimGAN_Refiner,
                                             mutate_def=BlockMutate_SimGAN,
                                             mate_def=BlockMate_WholeOnly_4Blocks
                                            )

        discriminator_def = self.construct_block_def(nickname = "discriminator_block",
                                             shape_def = BlockShapeMeta_SimGAN_Network, 
                                             operator_def = BlockOperators_SimGAN_Discriminator, 
                                             argument_def = BlockArguments_SimGAN_Discriminator,
                                             evaluate_def = BlockEvaluate_SimGAN_Discriminator,
                                             mutate_def=BlockMutate_SimGAN,
                                             mate_def=BlockMate_WholeOnly_4Blocks
                                            )

        train_config_def = self.construct_block_def(nickname = "train_config",
                                             shape_def = BlockShapeMeta_SimGAN_Train_Config, 
                                             operator_def = BlockOperators_SimGAN_Train_Config, 
                                             argument_def = BlockArguments_SimGAN_Train_Config,
                                             evaluate_def = BlockEvaluate_SimGAN_Train_Config,
                                             mutate_def=BlockMutate_SimGAN,
                                             mate_def=BlockMate_WholeOnly_4Blocks
                                            )

        self.construct_individual_def(block_defs = [refiner_def, discriminator_def, train_config_def],
                                      mutate_def = IndividualMutate_RollOnEachBlock,
                                      mate_def = IndividualMate_RollOnEachBlock,
                                      evaluate_def = IndividualEvaluate_SimGAN
                                      )

        # where to put this?
        self.construct_dataset()


    def construct_dataset(self):
        '''
        Constructs a train and validation 1D signal datasets
        '''
        # Can configure the real and simulated sizes + batch size, but we will use default
        self.training_datalist = [simganData.SimGANDataset(real_size=128**2, sim_size=256, batch_size=128)]
        self.validating_datalist = [simganData.SimGANDataset(real_size=int((128**2)/4), sim_size=128, batch_size=128)]
        # import pdb; pdb.set_trace()

    def objective_functions(self, population):
        '''
        Get the best refiner and discriminator from each individual in the population and do a tournament selection to rate them
        # TODO: add in the support size as a metric
        '''
        n_individuals = len(population.population)
        refiners = []
        discriminators = []
        indiv_inds = []
        bad_indiv_inds = []
        # import pdb; pdb.set_trace()
        for i, indiv in enumerate(population.population):
            if indiv.dead:
                indiv.fitness.values = (np.inf,)
            else:
                indiv_inds.append(i)
                R, D = indiv.output
                refiners.append(R.cpu())
                discriminators.append(D.cpu())
        
        # Run tournament and add ratings
        if len(refiners) > 0:
            refiner_ratings, _ = get_graph_ratings(refiners, discriminators, self.validating_datalist[0], 'cpu')
        
        for refiner_rating, ind in zip(refiner_ratings['r'].to_numpy(), indiv_inds):
            population.population[ind].fitness.values = (-1 * refiner_rating,) # Use negative ratings because ezCGP does minimization


    def check_convergence(self, universe):
        '''
        TODO: add code for determining whether convergence has been reached
        TODO: figure out if any of the below code is useful
        '''
        GENERATION_LIMIT = 3 #1000
        # SCORE_MIN = 1e-1

        # # only going to look at the first objective value which is rmse
        # # CAREFUL, after we added the ids, the values are now strings not floats
        # min_firstobjective_index = universe.pop_fitness_scores[:,0].astype(float).argmin()
        # min_firstobjective = universe.pop_fitness_scores[min_firstobjective_index,:-1].astype(float)
        # logging.warning("Checking Convergence - generation %i, best score: %s" % (universe.generation, min_firstobjective))

        # if universe.generation >= GENERATION_LIMIT:
        #     logging.warning("TERMINATING...reached generation limit.")
        #     universe.converged = True
        # if min_firstobjective[0] < SCORE_MIN:
        #     logging.warning("TERMINATING...reached minimum scores.")
        #     universe.converged = True


    def postprocess_generation(self, universe):
        '''
        TODO: add code for generation postprocessing
        TODO: figure out if any of the below code is useful
        '''
        # logging.info("Post Processing Generation Run")
        # save_things.save_fitness_scores(universe)

        # ith_indiv, _ = self.get_best_indiv(universe, ith_obj=0)
        # best_indiv = universe.population.population[ith_indiv]
        # active_count = len(best_indiv[0].active_nodes) - self.indiv_def[0].input_count - self.indiv_def[0].output_count
        # if hasattr(self, 'roddcustom_bestindiv'):
        #     self.roddcustom_bestindiv.append(best_indiv.id)
        #     self.roddcustom_bestscore.append(best_indiv.fitness.values)
        #     self.roddcustom_bestactive.append(active_count)
        # else:
        #     self.roddcustom_bestindiv = [best_indiv.id]
        #     self.roddcustom_bestscore = [best_indiv.fitness.values]
        #     self.roddcustom_bestactive = [active_count]

        # fig, axes = plot_things.plot_init(nrow=2, ncol=1, figsize=(15,10), ylim=(0,self.train_data.y[0].max()*1.25)) #axes always 2dim
        # plot_things.plot_regression(axes[0,0], best_indiv, self)
        # plot_things.plot_gaussian(axes[1,0], best_indiv, self)
        # plot_things.plot_legend()
        # plot_things.plot_save(fig, name=os.path.join(universe.output_folder, "gen%04d_bestindv.jpg" % universe.generation))


    def postprocess_universe(self, universe):
        '''
        TODO: add code for universe postprocessing
        TODO: figure out if any of the below code is useful
        '''
        # logging.info("Post Processing Universe Run")
        # save_things.save_population(universe)
        # save_things.save_population_asLisp(universe, self.indiv_def)

        # best_ids = np.array(self.roddcustom_bestindiv)
        # best_scores = np.array(self.roddcustom_bestscore)
        # best_activecount = np.array(self.roddcustom_bestactive)
        # # YO active nodes includes outputs and input nodes so 10 main nodes + 2 inputs + 1 output   
        # output_best_file = os.path.join(universe.output_folder, "custom_stats.npz")
        # np.savez(output_best_file, ids=best_ids,
        #                            scores=best_scores,
        #                            active_count=best_activecount,
        #                            genome_size=np.array([self.indiv_def[0].main_count]))
        # # i guess i want to save all the roddcustom_ attributes
        # # then open all the values for all the universes for each of the different runs
        # # and plot the different number of genomes in one color

        # # shoot...if doing more than one universe, need to delete these
        # self.roddcustom_bestindiv = []
        # self.roddcustom_bestscore = []
        # self.roddcustom_bestactive = []


    # TODO: see if anything here is useful for visualizaing simgans
    # def plot_custom_stats(self, folders):
    #     import glob
    #     import matplotlib.pyplot as plt

    #     if (type(folders) is str) and (os.path.isdir(folders)):
    #         '''# then assume we are looking for folders within this single folder
    #         poss_folders = os.listdir(folders)
    #         folders = []
    #         for poss in poss_folders:
    #             if os.path.isdir(poss):
    #                 folders.append(poss)'''
    #         # now that we are using glob below, we are all good...just make this into a list
    #         folders = [folders]
    #     elif type(folders) is list:
    #         # then continue as is
    #         pass
    #     else:
    #         print("we don't know how to handle type %s yet" % (type(folders)))

    #     # now try to find 'custom_stats.npz' in the folders
    #     stats = {}
    #     for folder in folders:
    #         npzs = glob.glob(os.path.join(folder,"*","custom_stats.npz"), recursive=True)
    #         for npz in npzs:
    #             data = np.load(npz)
    #             genome_size = data['genome_size'][0]
    #             if genome_size not in stats:
    #                 stats[genome_size] = {'ids': [],
    #                                       'scores': [],
    #                                       'active_count': []}
    #             for key in ['ids','scores','active_count']:
    #                 stats[genome_size][key].append(data[key])

    #     # now go plot
    #     #plt.figure(figsize=(15,10))
    #     matplotlib_colors = ['b','g','r','c','m','y']
    #     fig, axes = plt.subplots(2, 1, figsize=(16,8))
    #     for ith_size, size in enumerate(stats.keys()):
    #         for row, key in enumerate(['scores','active_count']):
    #             datas = stats[size][key]
    #             for data in datas:
    #                 if key == 'scores':
    #                     data = data[:,0]
    #                 axes[row].plot(data, color=matplotlib_colors[ith_size], linestyle="-", alpha=0.5)

    #     plt.show()
    #     import pdb; pdb.set_trace()
    #     plt.close()
