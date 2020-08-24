'''
root/problems/problem_multiGaussian.py
'''

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
from data.data_tools import data_loader
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
        super().__init__(population_size, number_universe, factory, mpi)

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
        self.data = data_loader.load_symbolicRegression([x, starting_sum], [y, noisy, goal_features])


    def objective_functions(self, indiv):
        if indiv.dead:
            indiv.fitness.values = (np.inf, np.inf, np.inf)
        else:
            clean_y, noisy_y, goal_features = self.data.y_train
            predict_y = indiv.output
            # how to extract the arguments to match to goal_features as well?
            error = clean_y-predict_y
            rms_error = np.sqrt(np.mean(np.square(error)))
            max_error = np.max(np.abs(error))
            # YO active nodes includes outputs and input nodes so 10 main nodes + 2 inputs + 1 output
            #active_error = np.abs(10+2+1-len(indiv[0].active_nodes)) #maybe cheating by knowing the goal amount ahead of time
            active_error = len(indiv[0].active_nodes)
            indiv.fitness.values = (rms_error, max_error, active_error)


    def check_convergence(self, universe):
        GENERATION_LIMIT = 10#00
        SCORE_MIN = 1e-1

        # only going to look at the first objective value which is rmse
        # CAREFUL, after we added the ids, the values are now strings not floats
        min_firstobjective_index = universe.fitness_scores[:,0].astype(float).argmin()
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

        ith_indiv, _ = self.get_best_indiv(universe, ith_obj=0)
        best_indiv = universe.population.population[ith_indiv]
        active_count = len(best_indiv[0].active_nodes) - self.indiv_def[0].input_count - self.indiv_def[0].output_count
        if hasattr(self, 'roddcustom_bestindiv'):
            self.roddcustom_bestindiv.append(best_indiv.id)
            self.roddcustom_bestscore.append(best_indiv.fitness.values)
            self.roddcustom_bestactive.append(active_count)
        else:
            self.roddcustom_bestindiv = [best_indiv.id]
            self.roddcustom_bestscore = [best_indiv.fitness.values]
            self.roddcustom_bestactive = [active_count]

        fig, axes = plot_things.plot_init(nrow=2, ncol=1, figsize=(15,10), ylim=(0,self.data.y_train[0].max()*1.25)) #axes always 2dim
        plot_things.plot_regression(axes[0,0], best_indiv, self)
        plot_things.plot_gaussian(axes[1,0], best_indiv, self)
        plot_things.plot_legend()
        plot_things.plot_save(fig, name=os.path.join(universe.output_folder, "gen%04d_bestindv.jpg" % universe.generation))


    def postprocess_universe(self, universe):
        '''
        save each individual at the end of the population
        '''
        logging.info("Post Processing Universe Run")
        save_things.save_population(universe)
        save_things.save_population_asLisp(universe, self.indiv_def)

        best_ids = np.array(self.roddcustom_bestindiv)
        best_scores = np.array(self.roddcustom_bestscore)
        best_activecount = np.array(self.roddcustom_bestactive)
        # YO active nodes includes outputs and input nodes so 10 main nodes + 2 inputs + 1 output   
        output_best_file = os.path.join(universe.output_folder, "custom_stats.npz")
        np.savez(output_best_file, ids=best_ids,
                                   scores=best_scores,
                                   active_count=best_activecount,
                                   genome_size=np.array([self.indiv_def[0].main_count]))
        # i guess i want to save all the roddcustom_ attributes
        # then open all the values for all the universes for each of the different runs
        # and plot the different number of genomes in one color

        # shoot...if doing more than one universe, need to delete these
        self.roddcustom_bestindiv = []
        self.roddcustom_bestscore = []
        self.roddcustom_bestactive = []


    def plot_custom_stats(self, folders):
        import glob
        import matplotlib.pyplot as plt

        if (type(folders) is str) and (os.path.isdir(folders)):
            '''# then assume we are looking for folders within this single folder
            poss_folders = os.listdir(folders)
            folders = []
            for poss in poss_folders:
                if os.path.isdir(poss):
                    folders.append(poss)'''
            # now that we are using glob below, we are all good...just make this into a list
            folders = [folders]
        elif type(folders) is list:
            # then continue as is
            pass
        else:
            print("we don't know how to handle type %s yet" % (type(folders)))

        # now try to find 'custom_stats.npz' in the folders
        stats = {}
        for folder in folders:
            npzs = glob.glob(os.path.join(folder,"*","custom_stats.npz"), recursive=True)
            for npz in npzs:
                data = np.load(npz)
                genome_size = data['genome_size'][0]
                if genome_size not in stats:
                    stats[genome_size] = {'ids': [],
                                          'scores': [],
                                          'active_count': []}
                for key in ['ids','scores','active_count']:
                    stats[genome_size][key].append(data[key])

        # now go plot
        #plt.figure(figsize=(15,10))
        matplotlib_colors = ['b','g','r','c','m','y']
        fig, axes = plt.subplots(2, 1, figsize=(16,8))
        for ith_size, size in enumerate(stats.keys()):
            for row, key in enumerate(['scores','active_count']):
                datas = stats[size][key]
                for data in datas:
                    if key is 'scores':
                        data = data[:,0]
                    axes[row].plot(data, color=matplotlib_colors[ith_size], linestyle="-", alpha=0.5)

        plt.show()
        import pdb; pdb.set_trace()
        plt.close()



