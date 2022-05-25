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
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
import multiprocessing

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

TIMESTEP = 0.5 # in seconds
BATCH_SIZES = [2**5, 2**6, 2**7, 2**8, 2**9, 2**10] # TODO
COLORS = ['k', 'b','g','r','c','m','y']


def get_gpu_mem_tf(return_dict):
    '''
    Assume that tf was already imported.
    
    TF allegedly allocates ALL available memory unless you set 'set_memory_growth' to True.
    So if we have an old version of TF and can't get gpu memory from tf directly, then we have to
    settle for the allocated memory instead of currently used memory and get it via nvidia-smi or something.
    UPDATE: memory growth set in problem_mnist.check_set_gpu(); do same in other problems, not here.
    '''
    memory_list = []
    return_dict['gpu_mem'] = memory_list

    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    total_memory = int(sp.check_output(command.split()).decode('ascii').split("\n")[1].split()[0])
    total_memory /= 1000 # convert MB to GB
    return_dict['gpu_total_mem'] = total_memory

    assert('tf' in globals()), "Using get_gpu_mem_tf but tensorflow as tf not imported"
    tf_version = float(".".join(tf.__version__.split(".")[:-1]))

    while True:
        if tf_version >= 2.5:
            # https://www.tensorflow.org/versions/r2.5/api_docs/python/tf/config/experimental/get_memory_info
            info = tf.config.experimental.get_memory_info('GPU:0')
            memory = gpu_info['current']
        elif tf_version >= 2.4:
            # https://www.tensorflow.org/versions/r2.4/api_docs/python/tf/config/experimental/get_memory_usage
            memory = tf.config.experimental.get_memory_usage('GPU:0')
        else:
            # https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow
            command = "nvidia-smi --query-gpu=memory.used --format=csv"
            used_memory = int(sp.check_output(command.split()).decode('ascii').split("\n")[1].split()[0])
            used_memory /= 1000 # convert MB to GB

            command = "nvidia-smi --query-gpu=memory.reserved --format=csv"
            reserved_memory = int(sp.check_output(command.split()).decode('ascii').split("\n")[1].split()[0])
            reserved_memory /= 1000 # convert MB to GB

            memory = used_memory + reserved_memory

        memory_list.append(memory)
        return_dict['gpu_mem'] = memory_list
        time.sleep(TIMESTEP)


def get_gpu_mem(module, return_dict):
    '''
    I thought multiprocessing Process would have same imports but it doesn't
    for some reason so having to re-load_NN_module.

    Using multiprocessing Manager dict to store values.
    NOTE we have to constantly overwrite the key/value for it to get stored properly
    ...can't just 'append' values to existing key/value
    '''
    load_NN_module(module)
    if module == 'tensorflow':
        assert('tf' in globals()), "Using get_gpu_mem_tf but tensorflow as tf not imported"
        get_gpu_mem_tf(return_dict)
    elif module == 'torch':
        raise Exception("Haven't built a method to get gpu memory for pytorch.")
    else:
        raise Exception("Haven't built a method to get gpu memory for given module: %s" % module)


def train(indiv_material, indiv_def, problem):
    indiv_def.evaluate(indiv_material, problem.training_datalist, problem.validating_datalist)


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
            
            '''###################### ORIGINAL#############################
            # threading stuff
            p = multiprocessing.Process(target=train, args=(indiv_material, indiv_def, problem))
            start_time = time.time()
            p.start()
            gpu_memory = []
            while p.is_alive():
                gpu_memory.append(get_gpu_mem(module))
                time.sleep(TIMESTEP)
            p.join()
            run_time = time.time() - start_time'''
            #################################################################
            # use mulltiprocessing.Manager to store values from gpu_mem as it's run
            assert('tf' in globals()), "Using get_gpu_mem_tf but tensorflow as tf not imported"
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            p = multiprocessing.Process(target=get_gpu_mem, args=(module, return_dict))
            p.start()
            start_time = time.time()
            train(indiv_material, indiv_def, problem)
            run_time = time.time() - start_time
            p.kill() # or terminate?
            gpu_memory = return_dict['gpu_mem']


            ###################################################################

            if 'num_params' not in stats:
                # fill with architecture stats:
                stats['num_params'] = 0
                stats['num_layers'] = 0
    
            # add stats:
            # TODO
            stats[batch_size][i] = {}
            stats[batch_size][i]['time'] = run_time # single float
            stats[batch_size][i]['gpu_mem'] = gpu_memory # array of floats at every X timestep
            stats[batch_size][i]['sys_mem'] = gpu_memory # array of floats at every X timestep
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
    times = np.array(times)

    x_pos = np.arange(len(bar_labels)) + 0.5
    time_avg = times.mean(axis=1)
    time_std = times.std(axis=1)
    axes.bar(x_pos, time_avg, yerr=time_std)
    axes.set_xticks(x_pos)
    axes.set_xticklabels(bar_labels) 

    fig.suptitle('%s\nTraining Time Stats' % (problem.__class__.__module__))
    axes.set_ylabel('Ave Time')
    axes.set_xlabel('Batch Size')
    plt.savefig("TimeStats.png")
    
    ### mem -> time vs mem. 1 color for each batch_size. alpha low for overlap
    fig, axes = plt.subplots(2, 1, sharex=True)
    for i, batch_size in enumerate(BATCH_SIZES):
        color = COLORS[i]
        for run in range(run_count):
            gpu_mem = stats[batch_size][run]['gpu_mem']
            sys_mem = stats[batch_size][run]['sys_mem']
            assert(len(gpu_mem)==len(sys_mem))
            times = np.arange(len(gpu_mem)) * TIMESTEP
            kwargs = {'linestyle': '-',
                      'marker': 'x',
                      'color': color,
                      'alpha': 0.4}
            if run == 0:
                kwargs['label'] = batch_size
            axes[0].plot(times, gpu_mem, **kwargs)
            axes[1].plot(times, sys_mem, **kwargs)
    
    fig.suptitle("%s\nMemory Stats" % (problem.__class__.__module__)) # TODO, how to get class name??
    axes[0].legend()
    axes[0].set_ylabel('GPU Mem (GB)')
    axes[1].set_ylabel('System Mem (GB)')
    axes[1].set_xlabel('Time (s)')
    plt.savefig("MemoryStats.png")



def load_NN_module(module):
    if module in ['tensorflow', 'tf', 'keras']:
        module = 'tensorflow'
        globals()['tf'] = __import__(module)
    elif module in ['pytorch', 'torch']:
        module = 'torch'
        globals()['torch'] = __import__(module)
    else:
        raise Exception("Didn't recognize command line argument for 'module': %s" % (args.module))
    globals()['module'] = module


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

    args = parser.parse_args()

    # figure out which problem py file to import
    if args.problem.endswith('.py'):
        problem_filename = args.problem[:-3]
    else:
        problem_filename = args.problem
    problem_module = __import__(problem_filename)
    problem = problem_module.Problem()
    
    # load in NN module
    load_NN_module(args.module)
    
    go(problem, module, 10)
