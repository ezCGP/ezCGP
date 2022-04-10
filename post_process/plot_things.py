'''
root/post_process/plot_things.py
'''

### packages
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import deap.tools
from scipy import stats 
### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from misc import fake_mixturegauss
from codes.utilities.custom_logging import ezLogging


matplotlib_colors = ['b','g','r','c','m','y'] # 6 colors and excludes black 'k'


def plot_init(nrow=1, ncol=1, figsize=None, xlim=None, ylim=None):
    '''
    always going to make sure axes has rank 2...no matter what!
    '''
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    if type(axes) != np.ndarray:
        axes = np.array([axes])
    axes = axes.reshape((nrow, ncol))
    for i in range(nrow):
        for j in range(ncol):
            if xlim is not None:
                axes[i,j].set_xlim(xlim)
            if ylim is not None:
                axes[i,j].set_ylim(ylim)
    return fig, axes


def label_axis(axis, x, y, fontsize=12, **kwargs):
    axis.set_xlabel(x, fontsize=fontsize, **kwargs)
    axis.set_ylabel(y, fontsize=fontsize, **kwargs)


def plot_legend(fig=None, axis=None):
    if fig is not None:
        fig.legend()
    elif axis is not None:
        axis.legend()
    else:
        plt.legend()


def square_figure(fig):
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()


def plot_save(fig, name):
    fig.savefig(name)
    plt.close()


def plot_regression(ax, indiv, problem):
    '''
    assume the figure has already been created and we are passing in the axes of subplot

    indiv has .output which should be the final regression
    '''
    ax.plot(problem.training_datalist[0].x, problem.training_datalist[0].y, linestyle='-', color='k', label="true values1")
    training_output, validating_output = indiv.output
    ax.plot(problem.training_datalist[0].x, training_output[0], linestyle='--', color='c', label="regression1")


def plot_gaussian(ax, indiv, problem):
    for i, args in enumerate(problem.training_datalist[0].goal_features): #goal_features
        peak, std, intensity, ybump = args
        curve = fake_mixturegauss.one_gauss(problem.training_datalist[0].x, peak, std, intensity, ybump)
        if i == 0:
            ax.plot(problem.training_datalist[0].x, curve, linestyle='-', color='k', label="true values", alpha=1)
        else:
            ax.plot(problem.training_datalist[0].x, curve, linestyle='-', color='k', alpha=1)

    i = 0
    for node in indiv[0].active_nodes:
        if (node<0) or (node>=problem.indiv_def[0].main_count):
            # input or output node
            continue
        indivargs = []
        for arg_index in indiv[0][node]['args']:
            indivargs.append(indiv[0].args[arg_index].value) #gotta do .value to get it as float
        peak, std, intensity = indivargs
        ybump = 0
        curve = fake_mixturegauss.one_gauss(problem.training_datalist[0].x, peak, std, intensity, ybump)
        if i == 0:
            ax.plot(problem.training_datalist[0].x, curve, linestyle='--', color='c', label="regression")
        else:
            ax.plot(problem.training_datalist[0].x, curve, linestyle='--', color='c')
        i+=1


def find_pareto(data,
                minimization=True,
                max_x=1,
                max_y=1):
    is_pareto = np.ones(data.shape[0], dtype = bool)
    for i, c in enumerate(data):
        # Keep any point with a lower cost
        if is_pareto[i]:
            if minimization:
                is_pareto[is_pareto] = np.any(data[is_pareto]<c, axis=1)
            else:
                # Maximization
                is_pareto[is_pareto] = np.any(data[is_pareto]>c, axis=1)
            # And keep self
            is_pareto[i] = True
    # Downsample from boolean array + sort
    pareto_data = data[is_pareto, :]
    pareto_data =  pareto_data[np.argsort(pareto_data[:,0])]

    # add 2 extreme performances
    if minimization:
        x_limit = [max_x, pareto_data[:,1].min()]
        y_limit = [pareto_data[:,0].min(), max_y]
    else:
        x_limit = [pareto_data[:,0].max(), 0]
        y_limit = [0, pareto_data[:,1].max()]
    pareto_data = np.vstack((y_limit, pareto_data))
    pareto_data = np.vstack(([0, max_y], pareto_data))
    pareto_data = np.vstack((pareto_data, x_limit))
    pareto_data = np.vstack((pareto_data, [max_x,0]))

    return pareto_data


def get_pareto_front(fitness_scores,
                     maximize_objectives_list,
                     x_objective_index=0, y_objective_index=1,
                     first_front_only=False):
    '''
    There is already a similar method in Population class but this method hacks it so we look at
    strictly 2 objectives and plot the pareto for only those individuals.
    '''
    from codes.genetic_material import IndividualMaterial
    class FakeIndividual(IndividualMaterial):
        '''
        warning: this will make ezLogging statements for set_id but hopefully won't be totally distracting in use
        '''
        def __init__(self, maximize_objectives_list):
            super().__init__(maximize_objectives_list)

        def set_id(self, _id=None):
            super().set_id(_id)
            add_fake = "fake_%s" % self.id
            super().set_id(add_fake)

    # if fitness_scores is actually a list of individuals then convert to np.array of scores
    if (isinstance(fitness_scores, list)) and (isinstance(fitness_scores[0], IndividualMaterial)):
        new_fitness_scores = []
        for indiv in fitness_scores:
            new_fitness_scores.append(indiv.fitness.values) # not weighted
        fitness_scores = np.array(new_fitness_scores)

    fake_pop = []
    adjusted_objective_list = [maximize_objectives_list[x_objective_index],
                               maximize_objectives_list[y_objective_index]]
    for score in fitness_scores:
        fake_indiv = FakeIndividual(adjusted_objective_list)
        fake_indiv.fitness.values = (score[x_objective_index],
                                     score[y_objective_index])
        fake_pop.append(fake_indiv)

    # get pareto
    return deap.tools.sortNondominated(fake_pop, k=len(fake_pop), first_front_only=first_front_only)


def calc_auc(maximize_objectives_list, fitness_scores):
    '''
    assume fitness socres is a numpy array of ALL the scores for some group...not just the pareto front
    since we need a way to define the 'edges' of the polynomial we draw to calc the area
    '''
    # first get pareto front
    pareto_fronts = get_pareto_front(fitness_scores,
                                     maximize_objectives_list,
                                     x_objective_index=0, y_objective_index=1,
                                     first_front_only=True)

    # get and sort
    pareto_scores = []
    for indiv in pareto_fronts[0]:
        pareto_scores.append(indiv.fitness.wvalues)
    pareto_scores = np.array(pareto_scores)
    pareto_scores =  pareto_scores[np.argsort(pareto_scores[:,0])]

    # get left border
    left_border = (fitness_scores[:,0].min(), pareto_scores[0,1])
    right_border = (pareto_scores[-1,0], fitness_scores[:,1].min())

    # combine together
    auc_scores = np.vstack((left_border, pareto_scores))
    auc_scores = np.vstack((auc_scores, right_border))

    # calc
    box_widths = np.diff(auc_scores[:-1,0])
    box_heights = pareto_scores[:,1] - right_border[1] #auc_scores[1:-1:,1] - auc_scores[-1,1] <- the same calc
    auc = np.sum(box_widths * box_heights)
    return auc


def calc_auc_multi_gen(maximize_objectives_list, *all_fitness_scores):
    '''
    basically the same as calc_auc but each generation uses the same left/right border values
    '''
    # calculate the borders
    extreme_left = np.inf
    extreme_right = np.inf
    for gen_scores in all_fitness_scores:
        extreme_left = min(extreme_left, gen_scores[:,0].min())
        extreme_right = min(extreme_right, gen_scores[:,1].min())
        print(extreme_left, extreme_right)

    # now get pareto and calc auc for each gen
    all_auc = []
    for gen_scores in all_fitness_scores:
        pareto_fronts = get_pareto_front(gen_scores,
                                         maximize_objectives_list,
                                         x_objective_index=0, y_objective_index=1,
                                         first_front_only=True)
        # get and sort
        pareto_scores = []
        for indiv in pareto_fronts[0]:
            pareto_scores.append(indiv.fitness.wvalues)
        pareto_scores = np.array(pareto_scores)
        pareto_scores =  pareto_scores[np.argsort(pareto_scores[:,0])]

        # get left border
        left_border = (extreme_left, pareto_scores[0,1])
        right_border = (pareto_scores[-1,0], extreme_right)

        # combine together
        auc_scores = np.vstack((left_border, pareto_scores))
        auc_scores = np.vstack((auc_scores, right_border))

        # calc
        box_widths = np.diff(auc_scores[:-1,0])
        box_heights = pareto_scores[:,1] - right_border[1] #auc_scores[1:-1:,1] - auc_scores[-1,1] <- the same calc
        auc = np.sum(box_widths * box_heights)
        all_auc.append(auc)

    return all_auc


def plot_pareto_front(axis,
                      fitness_scores,
                      minimization=True,
                      color='c', label='',
                      x_objective_index=0, y_objective_index=1,
                      objective_names=None,
                      maximize_objectives=None,
                      max_x=1, max_y=1):
    '''
    fitness_scores -> np array of shape (population size before population selection, number of objective scores defined in problem())
    x/y_objective_index -> say we have 4 objective scores, which of the 4 should be plotted on the x-axis and which on the y-axis...by index
    '''
    # reduce fitness scores to the 2 objectives we care about
    
    fitness_scores = fitness_scores[:, [x_objective_index,y_objective_index]]
    
    #remove inf values
    mask = np.all(np.isnan(fitness_scores) | np.isinf(fitness_scores), axis=1)
    fitness_scores = fitness_scores[~mask]

    #Convert max objectives to min (*-1 to points) so that we can make nice plot
    max_obj = [maximize_objectives[x_objective_index], 
               maximize_objectives[y_objective_index]]
   
    labels = [objective_names[x_objective_index], 
              objective_names[y_objective_index]]

    for i, obj in enumerate(max_obj):
        if obj == True:
            fitness_scores[:, i] *= -1
            labels[i] = labels[i] + ' (negated)'

    if max_x is None:
        max_x = fitness_scores[:,0].max()
    if max_y is None:
        max_y = fitness_scores[:,1].max()

    # reduce to pareto
    pareto_scores = find_pareto(fitness_scores, minimization, max_x, max_y)
    # Plot Pareto steps.
    if minimization:
        where = 'post'
    else:
        where = 'pre'
    axis.step(pareto_scores[:,0], pareto_scores[:,1], where=where, color=color, label=label, marker="*")
    ''' in the legend, change the line to a rectangle block; you will have to remove label from the axis.step() call as well
    red_patch = mpatches.Patch(color=color, label=label)
    axis.legend(handles=[red_patch])
    '''

    axis.set_xlim(0,max_x)
    axis.set_ylim(0,max_y)
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    
    auc = np.sum(np.diff(pareto_scores[:,0])*pareto_scores[0:-1,1])
    axis.set_title(f"Pareto Front AUC: {auc}")
    # Calculate the Area under the Curve as a Riemann sum
    return auc


def plot_pareto_front2(axis,
                       pareto_fronts,
                       color=None, label='',
                       x_objective_index=0, y_objective_index=1,
                       xlabel=None, ylabel=None,
                       min_x=0, max_x=1,
                       min_y=0, max_y=1):
    '''
    New method now that we have a method in Population class to get the pareto fronts from hall of fame.
    pareto_fronts should be a list of lists as returned by the Population.get_pareto_front() method.

    Note that the Fitness class for each individual has a weighted values attribute which takes into
    account whether we are maximizing or minimizing
    
    x/y_objective_index -> say we have 4 objective scores, which of the 4 should be plotted on the x-axis and which on the y-axis...by index
    '''
    sofar_min_x = np.inf
    sofar_max_x = -np.inf
    sofar_min_y = np.inf
    sofar_max_y = -np.inf

    pseudo_ninf = -1*int(1e12-1)

    for c, front in enumerate(pareto_fronts):
        # grab scores for front
        fitness_scores = []
        for indiv in front:
            fitness_scores.append([indiv.fitness.wvalues[x_objective_index],
                                   indiv.fitness.wvalues[y_objective_index]])
        fitness_scores = np.array(fitness_scores)

        # sort scores
        fitness_scores =  fitness_scores[np.argsort(fitness_scores[:,0])]

        # calculate before adding 'extreme points'
        #...will use these to help set plt axis limits if not provided
        sofar_min_x = min(sofar_min_x, fitness_scores[:,0].min())
        sofar_max_x = max(sofar_max_x, fitness_scores[:,0].max())
        sofar_min_y = min(sofar_min_y, fitness_scores[:,1].min())
        sofar_max_y = max(sofar_max_y, fitness_scores[:,1].max())

        # add 2 extreme performances
        yaxis_limit = (pseudo_ninf, fitness_scores[0,1])
        xaxis_limit = (fitness_scores[-1,0], pseudo_ninf)

        # combine together
        fitness_scores = np.vstack((yaxis_limit, fitness_scores))
        fitness_scores = np.vstack((fitness_scores, xaxis_limit))

        # get color
        if color is None:
            ith_color = c%len(matplotlib_colors)
            use_color = matplotlib_colors[ith_color]
        else:
            use_color = color

        # Plot Pareto steps.
        # NOTE: if minimizing, we would use where='pre' post
        axis.step(fitness_scores[:,0], fitness_scores[:,1], where='pre', color=color, label=label, marker="*")
        ''' in the legend, change the line to a rectangle block; you will have to remove label from the axis.step() call as well
        red_patch = mpatches.Patch(color=color, label=label)
        axis.legend(handles=[red_patch])
        '''

    def get_limits(i, min_axis, max_axis, sofar_min, sofar_max):
        # going to assume that the data is sorted so not going to call min() or max()
        # that will also help avoid extreme points if added
        if min_axis is None:
            min_axis = sofar_min
            if min_axis < 0:
                min_axis *= 1.1
            else:
                min_axis *= 0.9

        if max_axis is None:
            max_axis = sofar_max
            if max_axis < 0:
                max_axis *= 0.9
            else:
                max_axis *= 1.1
        
        return min_axis, max_axis

    min_x, max_x = get_limits(0, min_x, max_x, sofar_min_x, sofar_max_x)
    min_y, max_y = get_limits(1, min_y, max_y, sofar_min_y, sofar_max_y)
    axis.set_xlim(min_x,max_x)
    axis.set_ylim(min_y,max_y)
    if xlabel is not None:
        axis.set_xlabel("%s" % xlabel)
    if ylabel is not None:
        axis.set_ylabel("%s" % ylabel)
    axis.set_title("Pareto Front")


def plot_pareto_front_from_fitness_npz(axis, population_fitness_npz, minimization, **kwargs):
    npz_values = np.load(population_fitness_npz)
    fitness_values = npz_values['fitness']
    if not minimization:
        # in ezcgp we minimize so if we want to maximize, then scores are negative
        # then undo the negation here
        fitness_values *= -1
    auc = plot_pareto_front(axis, fitness_values, minimization, **kwargs)

def plot_pareto_front_from_fitness_npz_all_generations(axis, output_folder, minimization, **kwargs):
    '''
    EXAMPLE
        fig, axes = plot_things.plot_init()
        plot_things.plot_pareto_front_from_fitness_npz_all_generations(axes[0,0],
                                                    universe.output_folder,
                                                    minimization=True,
                                                    max_x=None,
                                                    max_y=None)
        plot_things.plot_legend(fig)
        plot_things.plot_save(fig, os.path.join(universe.output_folder, "ting.jpg"))
    '''
    all_npzs = sorted(glob.glob(os.path.join(output_folder, "gen*_fitness.npz")))
    if len(all_npzs)==0:
        print("couldn't find any fitness npzs")

    for i, npz in enumerate(all_npzs):
        generation = int(os.path.basename(npz)[3:7]) # gen \d\d\d\d _fitness.npz
        plot_pareto_front_from_fitness_npz(axis, npz, minimization,
                                           color=matplotlib_colors[i],
                                           label="gen%04d" % generation,
                                           **kwargs)


def plot_fitness_over_time(axis, output_folder, minimization, objective_index=0, **kwargs):
    '''
    assume fitness_scores is an array of shape (pop size, num objectives),
    objective_index gives which of the 'num objectives' to use for plotting on yaxis.
    will get fitness_scores from saved npz files.
    generation number will be on xaxis.

    EXAMPLE
        fig, axes = plot_things.plot_init()
        plot_things.plot_fitness_over_time(axes[0,0], universe.output_folder, minimization=True, objective_index=0)
        plot_things.plot_save(fig, os.path.join(universe.output_folder, "ting2.jpg"))
    '''    
    all_npzs = sorted(glob.glob(os.path.join(output_folder, "gen*_fitness.npz")))
    if len(all_npzs)==0:
        print("couldn't find any fitness npzs")
        import pdb; pdb.set_trace()

    x_values = []
    y_values = []
    for npz in all_npzs:
        npz_values = np.load(npz)
        fitness_values = npz_values['fitness'][:,objective_index]
        if minimization:
            best_score = fitness_values.min()
        else:
            best_score = fitness_values.max()
        y_values.append(best_score)

        generation = int(os.path.basename(npz)[3:7]) # gen \d\d\d\d _fitness.npz
        x_values.append(generation)

    axis.plot(x_values, y_values, marker='*', **kwargs)
    axis.set_title("Fitness Over Generations")
    # force x-axis to be only integers
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    #import pdb; pdb.set_trace()


def draw_genome(universe, problem, individual_material, draw_inactive_nodes=False):
    ezLogging.info("Drawing Individual %s" % individual_material.id)
    from codes.utilities.visualize import Visualizer
    viz = Visualizer(universe.output_folder)
    filename = "gen_%04d_indiv_%s.pkl" % (universe.generation, individual_material.id)
    viz.visualize(individual_material, problem.indiv_def, filename, draw_inactive_nodes)

def plot_mse_metric(mse, fitness_scores, objective_names=None, maximize_objectives=None, fitness_index=None, save_path=None):
    # reduce fitness scores to the 2 objectives we care about
    fitness_scores = fitness_scores[:, fitness_index]
    
    #remove inf values
    mask = np.all(np.isnan(fitness_scores) | np.isinf(fitness_scores))
    fitness_scores = fitness_scores[~mask]

    #Convert max objectives to min (*-1 to points) so that we can make nice plot
    max_obj = maximize_objectives[fitness_index] 
   
    labels = objective_names[fitness_index] 
    if max_obj == True:
        fitness_scores *= -1
        labels = labels + ' (negated)'
    
    plt.figure(figsize=(19, 26))

    slope, intercept, r, p, se = stats.linregress(fitness_scores, mse)
    r = r * r
    plt.scatter(fitness_scores, mse) 
    plt.plot(fitness_scores, intercept + slope * np.array(fitness_scores),
    label=f"fitted line, m = {slope}, b = {intercept}, r^2 = {r}, p = {p}",
    )
    plt.legend()
    plt.tight_layout()
    plt.xlabel(labels)
    plt.ylabel('MSE')
    plt.title(f"{labels} vs MSE")
    plt.savefig(os.path.join(save_path, f"{labels}_vs_mse.png"))
