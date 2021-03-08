'''
root/post_process/plot_things.py
'''

### packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from misc import fake_mixturegauss


matplotlib_colors = ['b','g','r','c','m','y'] # 6 colors and excludes black 'k'


def plot_init(nrow=1, ncol=1, figsize=None, xlim=None, ylim=None):
    '''
    always going to make sure axes has rank 2...no matter what!
    '''
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = axes.reshape((nrow, ncol))
    for i in range(nrow):
        for j in range(ncol):
            if xlim is not None:
                axes[i,j].set_xlim(xlim)
            if ylim is not None:
                axes[i,j].set_ylim(ylim)
    return fig, axes


def plot_legend(fig=None):
    if fig is not None:
        fig.legend()
    else:
        plt.legend()


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


def find_pareto(data, miniminzation=True):
    is_pareto = np.ones(data.shape[0], dtype = bool)
    for i, c in enumerate(data):
        # Keep any point with a lower cost
        if is_pareto[i]:
            if miniminzation:
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
    if miniminzation:
        x_limit = [1, pareto_data[:,1].min()]
        y_limit = [pareto_data[:,0].min(), 1]
    else:
        x_limit = [pareto_data[:,0].max(), 0]
        y_limit = [0, pareto_data[:,1].max()]
    pareto_data = np.vstack((y_limit, pareto_data))
    pareto_data = np.vstack((pareto_data, x_limit))

    return pareto_data


def plot_pareto_front_from_fitness_npz(axis, population_fitness_npz, minimization, **kwargs):
    npz_values = np.load(population_fitness_npz)
    fitness_values = npz_values['fitness']
    if not minimization:
        # in ezcgp we minimize so if we want to maximize, then scores are negative
        # then undo the negation here
        fitness_scores *= -1
    plot_pareto_front(axis, fitness_values, **kwargs)


def plot_pareto_front(axis,
                      fitness_scores,
                      minimization=True,
                      color='c', label='',
                      x_objective_index=0, y_objective_index=1):
    '''
    fitness_scores -> np array of shape (population size before population selection, number of objective scores defined in problem())
    x/y_objective_index -> say we have 4 objective scores, which of the 4 should be plotted on the x-axis and which on the y-axis...by index
    '''
    # redcue fitness scores to the 2 objectives we care about
    fitness_scores = fitness_scores[:, [x_objective_index,y_objective_index]]

    # reduce to pareto
    pareto_scores = find_pareto(fitness_scores, minimization)
    # Plot Pareto steps.
    if minimization:
        where = 'post'
    else:
        where = 'pre'
    axis.step(pareto_scores[:,0], pareto_scores[:,1], where=where, color=color, label=label)
    ''' in the legend, change the line to a rectangle block; you will have to remove label from the axis.step() call as well
    red_patch = mpatches.Patch(color=color, label=label)
    axis.legend(handles=[red_patch])
    '''

    # Calculate the Area under the Curve as a Riemann sum
    auc = np.sum(np.diff(pareto_scores[:,0])*pareto_scores[0:-1,1])
    return auc