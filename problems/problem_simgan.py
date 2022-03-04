### packages
import os
import numpy as np
import logging
import torch
import pickle as pkl
from copy import deepcopy

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems.problem_definition import ProblemDefinition_Abstract, welless_check_decorator
from codes.factory import Factory_SimGAN
from data.data_tools import simganData
from codes.utilities.custom_logging import ezLogging
from codes.utilities.gan_tournament_selection import get_graph_ratings
from codes.utilities.simgan_fid_metric import get_fid_scores
from codes.block_definitions.shapemeta.block_shapemeta import BlockShapeMeta_SimGAN_Network, BlockShapeMeta_SimGAN_Train_Config
from codes.block_definitions.operators.block_operators import BlockOperators_SimGAN_Refiner, BlockOperators_SimGAN_Discriminator, BlockOperators_SimGAN_Train_Config
from codes.block_definitions.arguments.block_arguments import BlockArguments_SimGAN_Refiner, BlockArguments_SimGAN_Discriminator, BlockArguments_SimGAN_Train_Config
from codes.block_definitions.evaluate.block_evaluate_pytorch import BlockEvaluate_SimGAN_Refiner, BlockEvaluate_SimGAN_Discriminator, BlockEvaluate_SimGAN_Train_Config 
from codes.block_definitions.mutate.block_mutate import BlockMutate_OptB_No_Single_Ftn, BlockMutate_OptB, BlockMutate_ArgsOnly
from codes.block_definitions.mate.block_mate import BlockMate_WholeOnly
from codes.individual_definitions.individual_mutate import IndividualMutate_RollOnEachBlock_LimitedMutants
from codes.individual_definitions.individual_mate import IndividualMate_RollOnEachBlock
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_SimGAN
from post_process import save_things
from post_process import plot_things


class Problem(ProblemDefinition_Abstract):
    '''
    Not intented to see if this does a good job at evolving but rather just a quick way to test out the different
    mating, mutating, operators etc with multiple blocks.
    '''
    def __init__(self):
        population_size = 4 #must be divisible by 4 if doing mating
        number_universe = 1
        factory = Factory_SimGAN
        mpi = False
        genome_seeds = [["misc/IndivSeed_SimGAN_Seed0/RefinerBlock_lisp.txt",
                         "misc/IndivSeed_SimGAN_Seed0/DiscriminatorBlock_lisp.txt",
                         "misc/IndivSeed_SimGAN_Seed0/ConfigBlock_lisp.txt"]]*population_size
        super().__init__(population_size, number_universe, factory, mpi, genome_seeds)
        self.relativeScoring = True # this will force universe to be instance of RelativePopulationUniverseDefinition() in main.py

        refiner_def = self.construct_block_def(nickname = "refiner_block",
                                               shape_def = BlockShapeMeta_SimGAN_Network, 
                                               operator_def = BlockOperators_SimGAN_Refiner, 
                                               argument_def = BlockArguments_SimGAN_Refiner,
                                               evaluate_def = BlockEvaluate_SimGAN_Refiner,
                                               mutate_def=BlockMutate_OptB_No_Single_Ftn(prob_mutate=0.2),
                                               mate_def=BlockMate_WholeOnly(prob_mutate=1/3)
                                              )

        discriminator_def = self.construct_block_def(nickname = "discriminator_block",
                                                     shape_def = BlockShapeMeta_SimGAN_Network, 
                                                     operator_def = BlockOperators_SimGAN_Discriminator, 
                                                     argument_def = BlockArguments_SimGAN_Discriminator,
                                                     evaluate_def = BlockEvaluate_SimGAN_Discriminator,
                                                     mutate_def=BlockMutate_OptB(prob_mutate=0.2),
                                                     mate_def=BlockMate_WholeOnly(prob_mutate=1/3)
                                                    )

        train_config_def = self.construct_block_def(nickname = "train_config",
                                                    shape_def = BlockShapeMeta_SimGAN_Train_Config, 
                                                    operator_def = BlockOperators_SimGAN_Train_Config, 
                                                    argument_def = BlockArguments_SimGAN_Train_Config,
                                                    evaluate_def = BlockEvaluate_SimGAN_Train_Config,
                                                    mutate_def=BlockMutate_ArgsOnly(prob_mutate=0.1),
                                                    mate_def=BlockMate_WholeOnly(prob_mutate=1/3)
                                                   )

        self.construct_individual_def(block_defs = [refiner_def, discriminator_def, train_config_def],
                                      mutate_def = IndividualMutate_RollOnEachBlock_LimitedMutants,
                                      mate_def = IndividualMate_RollOnEachBlock,
                                      evaluate_def = IndividualEvaluate_SimGAN
                                      )
        self.construct_dataset()


    def construct_dataset(self):
        '''
        Constructs a train and validation 1D signal datasets
        '''
        # Can configure the real and simulated sizes + batch size, but we will use default
        train_config_dict = {'device': 'cuda'} # was gpu but that didn't work anymore
        self.training_datalist = [simganData.SimGANDataset(real_size=512, sim_size=128**2, batch_size=128),
                                  train_config_dict]
        self.validating_datalist = [simganData.SimGANDataset(real_size=128, sim_size=int((128**2)/4), batch_size=128)]


    def set_optimization_goals(self):
        self.maximize_objectives = [True, False]


    @welless_check_decorator
    def objective_functions(self, population):
        '''
        Get the best refiner and discriminator from each individual in the population and do a tournament selection to rate them
        # TODO: add in the support size as a metric
        '''
        n_individuals = len(population.population)
        refiners = []
        discriminators = []
        alive_individual_index = []
        for i, indiv in enumerate(population.population):
            if not indiv.dead:
                alive_individual_index.append(i)
                R, D = indiv.output
                refiners.append(R.cpu())
                discriminators.append(D.cpu())
        
        # Run tournament and add ratings
        if len(refiners) > 0:
            refiner_ratings, _ = get_graph_ratings(refiners,
                                                   discriminators,
                                                   self.validating_datalist[0],
                                                   'cpu')
            refiner_fids = get_fid_scores(refiners, self.validating_datalist[0]) 
        
        for indx, rating, fid in zip(alive_individual_index,
                                     refiner_ratings['r'],
                                     refiner_fids):
            population.population[indx].fitness.values = (rating, fid)


    def check_convergence(self, universe):
        '''
        TODO: add code for determining whether convergence has been reached
        '''
        GENERATION_LIMIT = 50 # TODO
        if universe.generation >= GENERATION_LIMIT:
            ezLogging.warning("TERMINATING...reached generation limit.")
            universe.converged = True


    def population_selection(self, universe):
        for i, indiv in enumerate(universe.population.population):
            ezLogging.info("Final Population Scores: (%i) %s %s" % (i, indiv.id, indiv.fitness.values))

        super().population_selection(universe)

        for i, indiv in enumerate(universe.population.population):
            ezLogging.info("Next Population Scores: (%i) %s %s" % (i, indiv.id, indiv.fitness.values))


    def save_pytorch_individual(self, universe, original_individual):
        '''
        can't use save_things.save_population() because can't pickle nn.Module,
        so we're going to save the block and individual outputs into a folder for each individual,
        then delete those outputs so we can use save_things.save_population() normally.
        '''
        ezLogging.debug("Saving individual %s from generation %i" % (original_individual.id, universe.generation))
        # deepcopy in-case we still plan on using individual for another evolution
        individual = deepcopy(original_individual)

        # handle file names and locations
        name = "gen_%04d_indiv_%s" % (universe.generation, individual.id)
        attachment_folder = os.path.join(universe.output_folder, name)
        os.makedirs(attachment_folder, exist_ok=False)

        # save models
        torch.save(individual[0].output.state_dict(),
                   os.path.join(attachment_folder, 'untrained_refiner'))
        torch.save(individual[1].output.state_dict(),
                   os.path.join(attachment_folder, 'untrained_discriminator'))
        torch.save(individual.output[0].state_dict(),
                   os.path.join(attachment_folder, 'trained_refiner'))
        torch.save(individual.output[1].state_dict(),
                   os.path.join(attachment_folder, 'trained_discriminator'))
        with open(os.path.join(attachment_folder, 'trainconfig_dict.pkl'), 'wb') as f:
            pkl.dump(individual[2].output, f)

        # now overwrite
        individual[0].output = []
        individual[1].output = []
        individual[2].output = []
        individual.output = []

        # save individual
        indiv_file = os.path.join(universe.output_folder, name+".pkl")
        with open(indiv_file, "wb") as f:
            pkl.dump(individual, f)


    def postprocess_generation(self, universe):
        '''
        Save fitness scores and the refiners on the pareto front of fitness scroes
        '''
        ezLogging.info("Post Processing Generation Run")
        save_things.save_fitness_scores(universe)
        
        for individual in universe.population.population:
            self.save_pytorch_individual(universe, individual)

        '''
        pareto_front = self.get_pareto_front(universe)
        for ind in pareto_front:
            indiv = universe.population.population[ind]
            ## Uncomment to take a look at refined waves
            # import pdb; pdb.set_trace()
            # R = indiv.output[0]
            # sim = self.training_datalist[0].simulated_raw[:16]
            # import torch 
            # refined = R(torch.Tensor(sim)).detach().cpu().numpy()

            # import matplotlib.pyplot as plt
            # sim_wave = sim[0][0]
            # plt.plot(sim_wave)
            # plt.show()
            # refined_wave = refined[0][0]
            # plt.plot(refined_wave)
            # plt.show()

            save_things.save_pytorch_model(universe, indiv.output[0], indiv.id + '-R') # Save refiner network and id
            # TODO: consider saving discriminator'''

        # TODO: consider plotting the pareto front/metrics for the population
        ezLogging.info(f"fitnesscores {universe.pop_fitness_scores}")
        fig, axes = plot_things.plot_init()
        auc = plot_things.plot_pareto_front(axes[0,0],
                                                    universe.pop_fitness_scores,
                                                    minimization=True,
                                                    objective_names=['Tournament', 'FID'],
                                                    maximize_objectives=self.maximize_objectives,
                                                    max_x=None,
                                                    max_y=None)
        plot_things.plot_legend(fig)
        plot_things.plot_save(fig, os.path.join(universe.output_folder, f"ting{universe.generation}.jpg"))
        ezLogging.info(f"AUC: {auc}")

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
