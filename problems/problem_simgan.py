### packages
import os
import numpy as np
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
import codes.utilities.simgan_feature_eval as feature_eval
from codes.utilities.simgan_fid_metric import get_fid_scores
from codes.utilities.simgan_support_size_eval import get_support_size
from codes.block_definitions.shapemeta.block_shapemeta import BlockShapeMeta_SimGAN_Network, BlockShapeMeta_SimGAN_Train_Config
from codes.block_definitions.operators.block_operators import BlockOperators_SimGAN_Refiner, BlockOperators_SimGAN_Discriminator, BlockOperators_SimGAN_Train_Config
from codes.block_definitions.arguments.block_arguments import BlockArguments_Auto
from codes.block_definitions.evaluate.block_evaluate_pytorch import BlockEvaluate_SimGAN_Refiner, BlockEvaluate_SimGAN_Discriminator, BlockEvaluate_SimGAN_Train_Config
from codes.block_definitions.mutate.block_mutate import BlockMutate_OptB_No_Single_Ftn, BlockMutate_OptB, BlockMutate_ArgsOnly
from codes.block_definitions.mate.block_mate import BlockMate_WholeOnly
from codes.individual_definitions.individual_mutate import IndividualMutate_RollOnEachBlock_LimitedMutants
from codes.individual_definitions.individual_mate import IndividualMate_RollOnEachBlock
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_SimGAN
from post_process import save_things
from post_process import plot_things
from post_process import plot_signals
from codes.utilities import decorators


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
        hall_of_fame_size = population_size*50
        super().__init__(population_size, number_universe, factory, mpi, genome_seeds, hall_of_fame_size)
        self.relativeScoring = True # this will force universe to be instance of RelativePopulationUniverseDefinition() in main.py

        refiner_def = self.construct_block_def(nickname = "refiner_block",
                                               shape_def = BlockShapeMeta_SimGAN_Network,
                                               operator_def = BlockOperators_SimGAN_Refiner,
                                               argument_def = BlockArguments_Auto(BlockOperators_SimGAN_Refiner().operator_dict, 10),
                                               evaluate_def = BlockEvaluate_SimGAN_Refiner,
                                               mutate_def=BlockMutate_OptB_No_Single_Ftn(prob_mutate=0.2, num_mutants=2),
                                               mate_def=BlockMate_WholeOnly(prob_mate=1/3)
                                              )

        discriminator_def = self.construct_block_def(nickname = "discriminator_block",
                                                     shape_def = BlockShapeMeta_SimGAN_Network,
                                                     operator_def = BlockOperators_SimGAN_Discriminator,
                                                     argument_def = BlockArguments_Auto(BlockOperators_SimGAN_Discriminator().operator_dict, 15),
                                                     evaluate_def = BlockEvaluate_SimGAN_Discriminator,
                                                     mutate_def=BlockMutate_OptB(prob_mutate=0.2, num_mutants=2),
                                                     mate_def=BlockMate_WholeOnly(prob_mate=1/3)
                                                    )

        train_config_def = self.construct_block_def(nickname = "train_config",
                                                    shape_def = BlockShapeMeta_SimGAN_Train_Config,
                                                    operator_def = BlockOperators_SimGAN_Train_Config,
                                                    argument_def = BlockArguments_Auto(BlockOperators_SimGAN_Train_Config().operator_dict, 10),
                                                    evaluate_def = BlockEvaluate_SimGAN_Train_Config,
                                                    mutate_def=BlockMutate_ArgsOnly(prob_mutate=0.1, num_mutants=2),
                                                    mate_def=BlockMate_WholeOnly(prob_mate=1/3)
                                                   )

        self.construct_individual_def(block_defs = [refiner_def, discriminator_def, train_config_def],
                                      mutate_def = IndividualMutate_RollOnEachBlock_LimitedMutants,
                                      mate_def = IndividualMate_RollOnEachBlock,
                                      evaluate_def = IndividualEvaluate_SimGAN
                                      )
        self.construct_dataset()


    @decorators.stopwatch_decorator
    def construct_dataset(self):
        '''
        Constructs a train and validation 1D signal datasets
        '''
        # Can configure the real and simulated sizes + batch size, but we will use default
        train_config_dict = {'device': 'cuda', # was gpu but that didn't work anymore
                             'offline_mode': False} # see Issue #268 to get pretrained models working offline
        self.training_datalist = [simganData.SimGANDataset(real_size=512, sim_size=128**2, batch_size=4),
                                  train_config_dict]
        self.validating_datalist = [simganData.SimGANDataset(real_size=128, sim_size=int((128**2)/4), batch_size=4)]


    def set_optimization_goals(self):
        self.maximize_objectives = [False, False, False, True]
        self.objective_names = ["FID", "KS stat", "Significant Count", "Avg Feature P-value"] # will be helpful for plotting later


    @decorators.stopwatch_decorator
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
        if len(alive_individual_index) > 0:
            #  Objective #1 - NO LONGER AN OBJECTIVE FOR POPULATION SELECTION
            ezLogging.info("Calculating Objective 1")
            refiner_ratings, _ = get_graph_ratings(refiners,
                                                   discriminators,
                                                   self.validating_datalist[0],
                                                   'cpu')

            #  Objective #2
            ezLogging.info("Calculating Objective 2")
            refiner_fids, mses = get_fid_scores(refiners, self.validating_datalist[0], offline_mode=self.training_datalist[1]['offline_mode'])
            #refiner_fids, mses = (np.random.random(size=len(refiners)), np.random.random(size=len(refiners))) #<-sometimes i get a gpu memory error on above step so i replace with this in testing

            # Objective #3, #4, #5
            ezLogging.info("Calculating Objective 3,4,5")
            refiner_feature_dist = feature_eval.calc_feature_distances(refiners, self.validating_datalist[0], 'cpu')

            # Objective #6, #7
            ezLogging.info("Calculating Objective 6,7")
            refiner_t_tests = feature_eval.calc_t_tests(refiners, self.validating_datalist[0], 'cpu')

            # Objective #8
            #ezLogging.info("Calculating Objective 8")
            #support_size = get_support_size(refiners, self.validating_datalist[0], 'cpu')


            for indx, rating, fid, kl_div, wasserstein_dist, ks_stat, num_sig, avg_feat_pval, mse \
                in zip(alive_individual_index,
                    refiner_ratings['r'],
                    refiner_fids,
                    refiner_feature_dist['kl_div'],
                    refiner_feature_dist['wasserstein_dist'],
                    refiner_feature_dist['ks_stat'],
                    refiner_t_tests['num_sig'],
                    refiner_t_tests['avg_feat_pval'],
                    mses):
                # since refiner rating is a 'relative' score, we are not going to set it to fitness value to be used in population selection
                # BUT we will keep it available as metadata
                if hasattr(population.population[indx], 'refiner_rating'):
                    population.population[indx].refiner_rating.append(rating)
                else:
                    population.population[indx].refiner_rating = [rating]

                # mse is used to eval eval functions, we are not going to set it to fitness value to be used in population selection
                # BUT we will keep it available as metadata
                if hasattr(population.population[indx], 'mse'):
                    population.population[indx].mse.append(mse)
                else:
                    population.population[indx].mse = [mse]
                
                # Issue 219 - filtering down to only 4 objectives:
                # fid (#2), ks_stat (#5), num_sig (#6), and avg_feat_pval (#7)
                population.population[indx].fitness.values = (fid, ks_stat, num_sig, avg_feat_pval)


    def check_convergence(self, universe):
        '''
        TODO: add code for determining whether convergence has been reached
        '''
        GENERATION_LIMIT = 1 # TODO
        if universe.generation >= GENERATION_LIMIT:
            ezLogging.warning("TERMINATING...reached generation limit.")
            universe.converged = True


    def population_selection(self, universe):
        for i, indiv in enumerate(universe.population.population):
            ezLogging.warning("Final Population Scores: (%i) %s %s" % (i, indiv.id, indiv.fitness.values))

        next_pop = super().population_selection(universe)

        for i, indiv in enumerate(next_pop):
            ezLogging.warning("Next Population Scores: (%i) %s %s" % (i, indiv.id, indiv.fitness.values))

        return next_pop


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
        # NOTE if indiv.dead then some of these values may not be filled
        if not individual[0].dead:
            torch.save(individual[0].output[0].state_dict(),
                       os.path.join(attachment_folder, 'untrained_refiner'))

        if not individual[1].dead:
            torch.save(individual[1].output[0].state_dict(),
                       os.path.join(attachment_folder, 'untrained_discriminator'))

        if not individual[2].dead:
            with open(os.path.join(attachment_folder, 'trainconfig_dict.pkl'), 'wb') as f:
                pkl.dump(individual[2].output, f)

        if not individual.dead:
            torch.save(individual.output[0].state_dict(),
                       os.path.join(attachment_folder, 'trained_refiner'))
            torch.save(individual.output[1].state_dict(),
                       os.path.join(attachment_folder, 'trained_discriminator'))

        # now overwrite
        individual[0].output = []
        individual[1].output = []
        individual[2].output = []
        individual.output = []
        individual.blocks[1].local_graph = None

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
        save_things.save_HOF_scores(universe)

        # to be used later to extract features
        # ...note that we allow features to be turned on/off in evolution but we still plot all features
        fe = feature_eval.FeatureExtractor()
        for individual in universe.population.population:
            if not individual.dead:
                self.save_pytorch_individual(universe, individual)
                plot_things.draw_genome(universe, self, individual)

                # the rest is just to plot signals
                num_signals = 5
                #sample_index_sim = np.random.choice(np.arange(len(self.validating_datalist[0].simulated_raw)), size=num_signals)
                sample_index_sim = np.arange(num_signals) #not letting it be random so we can easily compare between refiners
                simulated_batch = torch.tensor(self.validating_datalist[0].simulated_raw[sample_index_sim], dtype=torch.float, device='cpu')
                sample_index_real = np.random.choice(np.arange(len(self.validating_datalist[0].real_raw)), size=num_signals)
                real_batch = torch.tensor(self.validating_datalist[0].real_raw[sample_index_real], dtype=torch.float, device='cpu')
                R, D = individual.output
                refined_sim_batch = R.cpu()(simulated_batch)
                refined_sim_preds = D.cpu()(refined_sim_batch)
                real_preds = D.cpu()(real_batch)
                attachment_folder = os.path.join(universe.output_folder, "gen_%04d_indiv_%s_signals.png" % (universe.generation, individual.id))
                plot_signals.generate_img_batch(simulated_batch.data.cpu(),
                                                refined_sim_batch.data.cpu(),
                                                real_batch.data.cpu(),
                                                attachment_folder,
                                                refined_sim_preds,
                                                real_preds)

                # now plot the feature distributions...but use full batch
                simulated_batch = torch.tensor(self.validating_datalist[0].simulated_raw, dtype=torch.float, device='cpu')
                real_batch = torch.tensor(self.validating_datalist[0].real_raw, dtype=torch.float, device='cpu')
                refined_sim_batch = R.cpu()(simulated_batch)

                simulated_features = fe.get_features(np.squeeze(simulated_batch.cpu().detach().numpy())).T
                refined_sim_features = fe.get_features(np.squeeze(refined_sim_batch.cpu().detach().numpy())).T
                real_features = fe.get_features(np.squeeze(real_batch.cpu().detach().numpy())).T

                # what is the shape returned of get_features()
                for ith_feature, feature_name in enumerate(fe.feature_names):
                    fig, axes = plot_things.plot_init(1, 1)
                    data = [simulated_features[:,ith_feature], refined_sim_features[:,ith_feature], real_features[:,ith_feature]]
                    labels = ["Simulated", "Refined Sim", "Real"]
                    plot_things.violin(axes[0,0], data, labels)
                    axes[0,0].set_title("%s feature distributions" % feature_name)
                    name = os.path.join(universe.output_folder, "gen_%04d_indiv_%s_%s_distribution.png" % (universe.generation, individual.id, feature_name))
                    plot_things.plot_save(fig, name)



        # Pareto Plot for each objective combo at current HOF:
        for i in range(len(self.maximize_objectives)-1):
            for j in range(i+1,len(self.maximize_objectives)):
                x_obj = self.objective_names[i]
                y_obj = self.objective_names[j]
                # Grab Pareto front and visualize secondary waveforms...do it for each combo of objectives
                pareto_fig, pareto_axis = plot_things.plot_init(nrow=1, ncol=1, figsize=None, xlim=None, ylim=None)
                pareto_fronts = plot_things.get_pareto_front(universe.population.hall_of_fame.items,
                                                             self.maximize_objectives,
                                                             x_objective_index=i,
                                                             y_objective_index=j,
                                                             first_front_only=False)
                plot_things.plot_pareto_front2(pareto_axis[0,0],
                                               pareto_fronts,
                                               color=None, label='',
                                               x_objective_index=0, y_objective_index=1,
                                               xlabel=x_obj, ylabel=y_obj,
                                               min_x=None, max_x=None,
                                               min_y=None, max_y=None)

                #plot_things.plot_legend(pareto_fig)
                plot_things.plot_save(pareto_fig,
                                      os.path.join(universe.output_folder,
                                                   "pareto_front_gen%04d_%s_vs_%s.png" % (universe.generation, x_obj, y_obj)))


        # Best Pareto Plot Over time
        for i in range(len(self.maximize_objectives)-1):
            for j in range(i+1,len(self.maximize_objectives)):
                x_obj = self.objective_names[i]
                y_obj = self.objective_names[j]

                pareto_fig, pareto_axis = plot_things.plot_init(nrow=1, ncol=1, figsize=None, xlim=None, ylim=None)
                for gen in range(universe.generation+1):
                    hof_fitness_file = os.path.join(universe.output_folder, "gen%04d_hof_fitness.npz" % gen)
                    hof_fitness = np.load(hof_fitness_file)['fitness']
                    pareto_fronts = plot_things.get_pareto_front(hof_fitness,
                                                                 self.maximize_objectives,
                                                                 x_objective_index=i,
                                                                 y_objective_index=j,
                                                                 first_front_only=True)
                    plot_things.plot_pareto_front2(pareto_axis[0,0],
                                                   pareto_fronts,
                                                   color=None, label="HOF Gen %i" % (gen),
                                                   x_objective_index=0, y_objective_index=1,
                                                   xlabel=x_obj, ylabel=y_obj,
                                                   min_x=None, max_x=None,
                                                   min_y=None, max_y=None)

                plot_things.plot_legend(pareto_fig)
                plot_things.plot_save(pareto_fig,
                                      os.path.join(universe.output_folder,
                                                   "pareto_front_overtime_gen%04d_%s_vs_%s.png" % (universe.generation, x_obj, y_obj)))


        # AUC over time:
        # get files:
        all_hof_scores = []
        for gen in range(universe.generation+1):
            hof_fitness_file = os.path.join(universe.output_folder, "gen%04d_hof_fitness.npz" % gen)
            hof_fitness = np.load(hof_fitness_file)['fitness']
            all_hof_scores.append(hof_fitness)

        # now for each combo of objectives, make a plot
        for i in range(len(self.maximize_objectives)-1):
            for j in range(i+1,len(self.maximize_objectives)):
                x_obj = self.objective_names[i]
                y_obj = self.objective_names[j]

                all_auc = plot_things.calc_auc_multi_gen(self.maximize_objectives, i, j, *all_hof_scores)
                auc_fig, auc_axis = plot_things.plot_init(nrow=1, ncol=1, figsize=None, xlim=None, ylim=None)
                auc_axis[0,0].plot(all_auc, marker='*')
                auc_axis[0,0].set_xlabel("ith Generation")
                auc_axis[0,0].set_title("AUC over time\n%s_vs_%s" % (x_obj, y_obj))
                plot_things.plot_save(auc_fig,
                                      os.path.join(universe.output_folder, "AUC_overtime_gen%04d_%s_vs_%s.png" % (universe.generation, x_obj, y_obj)))


    def postprocess_universe(self, universe):
        '''
        TODO: add code for universe postprocessing
        '''
        # ezLogging.info("Post Processing Universe Run")
        # save_things.save_population(universe)
        # save_things.save_population_asLisp(universe, self.indiv_def)
        pass
