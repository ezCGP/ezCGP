'''
root/codes/utilities/train_network_stats.py

RE Issue #277 TODO
We want a way to train a given network and monitor stats
so that we can make more informed decision on batch size etc.

Assumptions we are going to make to get this working without adding
crazy complications:
* We are only going to look at 1 individual in the population (always the 0th index)
* Going to assume we only have 1 block in the indiv_def and that
* That batch size is a param in indiv_def[0] as set by BlockShapeMeta
'''

### packages
import os
import time
import gc
from copy import deepcopy
import matplotlib.pyplot as plt

### sys relative to root dir
import sys
from os.path import dirname, realpath, join
sys.path.append(dirname(dirname(dirname(realpath(__file__))))) # add root ezcgp folder to sys path
sys.path.append(join(dirname(dirname(dirname(realpath(__file__)))), "problems")) # add problems folder to sys path

### absolute imports wrt root





'''
kind of stats we want:
* gpu mem
* gpu core usage
* computer mem
* number of trainable parameters
* number of layers?
* batch size
* time it takes to go through all batches ie 1 epoch

we want to find correlation between variables we can change (batch size, network size)
vs outcome (time, mem usage)
'''

TIMESTEP = 5 # in seconds
BATCH_SIZES = [2**5, 2**6, 2**7, 2**8, 2**9, 2**10] # TODO
COLORS = ['k', 'b','g','r','c','m','y']

def go(problem, module, run_count=20):
    '''
    TODO
    '''
    factory = problem.Factory()
    # we only want to evaluate 1 individual so pick the first
    if len(problem.genome_seeds) > 1:
        problem.genome_seeds = [problem.genome_seeds[0]]
    population = factory.build_population(problem, 1, 0, 1)
    indiv_material = population[0]
    indiv_def = problem.indiv_def
    
    # TODO: should i set num epochs to 1? what happens when we change it to 5 or something?
    if hasattr(indiv_def[0], 'epochs'):
        indiv_def[0].epochs = 1
    else:
        raise Exception("Couldn't find 'batch_size' attr for block def; why isn't it in BlockShapeMeta?")    
    
    stats = {}
    for batch_size in BATCH_SIZES:
        stats[batch_size] = {}
        
        if hasattr(indiv_def[0], 'batch_size'):
            indiv_def[0].batch_size = batch_size
        else:
            raise Exception("Couldn't find 'batch_size' attr for block def; why isn't it in BlockShapeMeta?")
        
        for i in range(run_count):
            # start subprocess to collect stats?
            # get help with
            
            indiv_def.evaluate(indiv_material, problem.training_datalist, problem.validating_datalist)
            
            if 'num_params' not in stats:
                # fill with architecture stats:
                stats['num_params'] = 0
                stats['num_layers'] = 0
    
            # add stats:
            # TODO
            stats[batch_size][i]['time'] = 0 # single float
            stats[batch_size][i]['gpu_mem'] = 0 # array of floats at every X timestep
            stats[batch_size][i]['sys_mem'] = 0 # array of floats at every X timestep
            # TODO, should gpu_mem and sys_mem be the same length for each run?, right? depends on how we call i guess
            
            
            # reset individual...in this case, it is convenient to treat as a mutated indiv
            indiv_def.postprocess_evolved_individual(indiv_material, 0)
            time.sleep(5)
            gc.collect()
            
        # TODO is there a way to try/catch when gpu is going to run out of memory instead of python crashing?
        # put it in a subprocess or something?
    
    
    ### ANALYSIS
    ### time -> barplot for each batch_size with whiskers
    fig, axes = plt.subplots(1, 1)
    bar_labels = []
    times = []
    for batch_size in BATCH_SIZES:
        bar_labels.append(str(batch_size))
        batch_times = []
        for run in range(run_count):
            batch_times.append(stats[batch_size][run]['time'])
        times.append(batch_times)

    axes.bar() # TODO

    axes.suptitle('%s\nTraining Time Stats' % (problem.__str__()))
    axes.set_ylabel('Ave Time')
    axes.set_xlabel('Batch Size')
    
    
    ### mem -> time vs mem. 1 color for each batch_size. alpha low for overlap
    fig, axes = plt.subplots(1, 2, sharex=True)
    for i, batch_size in enumerate(BATCH_SIZES):
        color = COLORS[i]
        for run in range(num_count):
            gpu_mem = stats[batch_size][run]['gpu_mem']
            sys_mem = stats[batch_size][run]['sys_mem']
            assert(len(gpu_mem)==len(sys_mem))
            time = np.arange(len(gpu_mem)) * TIMESTEP
            kwargs = {'linestyle': '-',
                      'color': color,
                      'alpha': 0.4}
            if run == 0:
                kwargs['label'] = batch_size
            axes[0].plot(time, gpu_mem, **kwargs)
            axes[1].plot(time, sys_mem, **kwargs)
    
    axes[0].suptitle("%s\nMemory Stats" % (problem.__str__())) # TODO, how to get class name??
    axes[0].legend()
    axes[0].set_ylabel('GPU Mem (GB)')
    axes[1].set_ylabel('System Mem (GB)')
    axes[1].set_xlabel('Time (s)')
    






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--problem",
                        type = str,
                        required = True,
                        help = "pick which problem class to import")
    parser.add_argument("-m", "--module",
                        type = str,
                        required = False,
                        default = 'tensorflow',
                        help = "the module type will determine how we train and collect some stats")
    
    # figure out which problem py file to import
    if args.problem.endswith('.py'):
        problem_filename = args.problem[:-3]
    else:
        problem_filename = args.problem
    problem_module = __import__(problem_filename)
    problem = problem_module.Problem()
    
    # load in NN module
    if args.module in ['tensorflow', 'tf', 'keras']:
        module = 'tensorflow'
        globals()['tf'] = __import__(module)
    elif args.module in ['pytorch', 'torch']:
        module = 'torch'
        globals()['torch'] = __import__(module)
    else:
        raise Exception("Didn't recognize command line argument for 'module': %s" % (args.module))
    
    go(problem, module, 2)