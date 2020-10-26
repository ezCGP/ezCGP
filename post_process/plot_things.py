'''
root/post_process/plot_things.py
'''

### packages
import matplotlib.pyplot as plt

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
    ax.plot(problem.data.x_train[0], problem.data.y_train[0], linestyle='-', color='k', label="true values1")
    ax.plot(problem.data.x_train[0], indiv.output[0], linestyle='--', color='c', label="regression1")


def plot_gaussian(ax, indiv, problem):

    for i, args in enumerate(problem.data.y_train[-1]): #goal_features
        peak, std, intensity, ybump = args
        curve = fake_mixturegauss.one_gauss(problem.data.x_train[0], peak, std, intensity, ybump)
        if i == 0:
            ax.plot(problem.data.x_train[0], curve, linestyle='-', color='k', label="true values", alpha=1)
        else:
            ax.plot(problem.data.x_train[0], curve, linestyle='-', color='k', alpha=1)

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
        curve = fake_mixturegauss.one_gauss(problem.data.x_train[0], peak, std, intensity, ybump)
        if i == 0:
            ax.plot(problem.data.x_train[0], curve, linestyle='--', color='c', label="regression")
        else:
            ax.plot(problem.data.x_train[0], curve, linestyle='--', color='c')
        i+=1
