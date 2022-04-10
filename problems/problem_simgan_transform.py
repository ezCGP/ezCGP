### packages
import os
import numpy as np
import logging
import torch
import pickle as pkl

### sys relative to root dir
import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems import problem_simgan
from data.data_tools import simganData
from copy import deepcopy

from problems.problem_definition import ProblemDefinition_Abstract, welless_check_decorator
from codes.utilities.custom_logging import ezLogging
from codes.utilities.gan_tournament_selection import get_graph_ratings
import codes.utilities.simgan_feature_eval as feature_eval
from codes.utilities.simgan_fid_metric import get_fid_scores
from codes.utilities.simgan_support_size_eval import get_support_size
from post_process import save_things
from post_process import plot_things
from post_process import plot_signals
from codes.utilities import decorators
import torch

class Problem(problem_simgan.Problem):
    """
    Basically the same as the other simgan problem but we want to use a different dataset.
    This allows us to toggle between them really easily.
    """
    def __init__(self):
        super().__init__()


    def construct_dataset(self):
        """
        Constructs a train and validation 1D signal datasets
        """
        # Can configure the real and simulated sizes + batch size, but we will use default
        train_config_dict = {"device": "cuda", 'offline_mode': False}  # was gpu but that didn't work anymore
        self.training_datalist = [simganData.TransformSimGANDataset(real_size=512, sim_size=128**2, batch_size=128),
                                  train_config_dict]
        self.validating_datalist = [simganData.TransformSimGANDataset(real_size=128, sim_size=int((128**2)/4), batch_size=128)]

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
            #refiner_fids = np.random.random(size=len(refiners)) #<-sometimes i get a gpu memory error on above step so i replace with this in testing

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
        GENERATION_LIMIT = 50 # TODO
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

        print(individual.output)

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

        fitlist = []
        mses=[]
        for individual in universe.population.population:
            if not individual.dead:
                self.save_pytorch_individual(universe, individual)
                plot_things.draw_genome(universe, self, individual)

                # the rest is just to plot signals
                num_signals = 5
                sample_index_sim = np.random.choice(np.arange(len(self.validating_datalist[0].simulated_raw)), size=num_signals)
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

                # print meta eval
                if hasattr(individual, 'mse'):
                    mses.append(individual.mse[-1])
                    fitlist.append(individual.fitness.values)
                    ezLogging.warning("Meta eval mse: %s fitness scores: %s" % (individual.mse, individual.fitness.values))
                else:
                    ezLogging.warning("No mse")
        mses = np.array(mses)
        fitlist = np.array(fitlist)
        for i in range(len(self.maximize_objectives)):
            plot_things.plot_mse_metric(mses, 
                                        fitlist, 
                                        objective_names=self.objective_names,
                                        maximize_objectives=self.maximize_objectives,
                                        fitness_index=i,
                                        save_path=universe.output_folder)


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
                                                   "pareto_front_gen%04d_%s_vs_%s.jpg" % (universe.generation, x_obj, y_obj)))



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
                                                   "pareto_front_overtime_gen%04d_%s_vs_%s.jpg" % (universe.generation, x_obj, y_obj)))


        # AUC over time:
        # get files:
        all_hof_scores = []
        for gen in range(universe.generation+1):
            hof_fitness_file = os.path.join(universe.output_folder, "gen%04d_hof_fitness.npz" % gen)
            hof_fitness = np.load(hof_fitness_file)['fitness']
            all_hof_scores.append(hof_fitness)

        all_auc = plot_things.calc_auc_multi_gen(self.maximize_objectives, *all_hof_scores)
        auc_fig, auc_axis = plot_things.plot_init(nrow=1, ncol=1, figsize=None, xlim=None, ylim=None)
        auc_axis[0,0].plot(all_auc, marker='*')
        auc_axis[0,0].set_xlabel("ith Generation")
        auc_axis[0,0].set_title("AUC over time")
        plot_things.plot_save(auc_fig,
                              os.path.join(universe.output_folder, "AUC_overtime_gen%04d.jpg" % (universe.generation)))

