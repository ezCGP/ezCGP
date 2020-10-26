'''
root/post_process/save_things.py
'''

### packages
import pickle as pkl
import numpy as np
import os

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
    np.savez(output_fitness_file, fitness=universe.fitness_scores)
    ezLogging.debug("saved scores for generation %i" % universe.generation)


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

