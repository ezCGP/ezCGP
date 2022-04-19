'''
root/post_process/save_things.py
'''

### packages
import pickle as pkl
import numpy as np
from copy import deepcopy
import os
import shutil
import torch

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from codes.utilities.custom_logging import ezLogging


def save_fitness_scores(universe):
    '''
    save scores as an npz...remember that the last item in each row is the individual id

    here is how to open an npz:
       ting = np.load(output_fitness_file)
       fitness_values = ting['fitness']
    '''
    output_fitness_file = os.path.join(universe.output_folder, "gen%04d_fitness.npz" % universe.generation)
    np.savez(output_fitness_file, fitness=universe.pop_fitness_scores)
    ezLogging.debug("saved scores for generation %i" % universe.generation)


def save_HOF_scores(universe):
    '''
    save scores as an npz...remember that the last item in each row is the individual id

    here is how to open an npz:
       ting = np.load(output_fitness_file)
       fitness_values = ting['fitness']
    '''
    # gotta get scores as np.array first
    hof_scores = []
    #pareto_front = universe.population.get_pareto_front(use_hall_of_fame=True, first_front_only=True)[0]
    #for indiv in pareto_front:
    for indiv in universe.population.hall_of_fame.items:
        hof_scores.append(indiv.fitness.wvalues) # <- used weighted!
    hof_scores = np.array(hof_scores)

    output_fitness_file = os.path.join(universe.output_folder, "gen%04d_hof_fitness.npz" % universe.generation)
    np.savez(output_fitness_file, fitness=hof_scores)
    ezLogging.debug("saved HOF scores for generation %i" % universe.generation)


def save_population(universe):
    '''
    save each individual_material as a pickle file named with it's id

    here is how to open a pickled file:
        from codes.genetic_material import IndividualMaterial
        with open(indiv_file, "rb") as f:
            indiv = pkl.load(f)
    '''
    ezLogging.debug("saved each individual from population for generation %i" % universe.generation)
    for indiv in universe.population.population:
        indiv_file = os.path.join(universe.output_folder, "gen_%04d_indiv_%s.pkl" % (universe.generation, indiv.id))
        with open(indiv_file, "wb") as f:
            pkl.dump(indiv, f)


def save_pytorch_model(universe, network, indiv_id):
    '''
    save a PyTorch neural network
    '''
    ezLogging.debug("saving pytorch model from population for generation %i" % universe.generation)
    path = os.path.join(universe.output_folder, "gen_%04d_id_%s.pkl" % (universe.generation, indiv_id))
    # was getting a pickling error if didn't do .state_dict()
    #...thanks https://github.com/pytorch/pytorch/issues/7545
    torch.save(network.state_dict(), path)

            
def save_population_asLisp(universe, indiv_definition):
    '''    
    each individual will have it's own .txt file
    the ith line will be the ith block's lisp
    block_definition has a get_lisp method, and factory should have a method to load an individual
    from a txt file of lisps or given the lisp strings directly for each block
    
    get_lisp() should return a list with length the number of outputs for each block (should only work for 1 output rn)
    '''
    ezLogging.debug("saved each individual from population as lisp-string for generation %i" % universe.generation)
    for indiv_material in universe.population.population:
        indiv_file = os.path.join(universe.output_folder, "gen_%04d_indiv_%s_lisp.txt" % (universe.generation, indiv_material.id))
        with open(indiv_file, "w") as f:
            for block_def, block_material in zip(indiv_definition, indiv_material):
                block_def.get_lisp(block_material)
                line = " ".join(block_material.lisp) # list with same length as number of block outputs, so we make into single string
                f.write("%s\n" % line)


def copy_paste_file(src, dst):
    '''
    copy+paste any file over to problem_output_directory.
    this way we know for sure which version of the problem file resulted in the output.
    '''
    shutil.copyfile(src, dst)
    ezLogging.debug("copied %s to %s" % (src, dst))


def pickle_dump_object(thing, dst):
    '''
    easy way to dump an object somewhere
    '''
    with open(dst, 'wb') as f:
        pkl.dump(thing, f)
    ezLogging.debug("pickled an object to %s" % dst)
