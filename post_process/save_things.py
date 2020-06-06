'''
root/post_process/save_things.py
'''

### packages
import logging
import pickle as pkl
import numpy as np
import os

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root


def save_fitness_scores(universe):
    '''
    save scores as an npz...remember that the last item in each row is the individual id

    here is how to open an npz:
       ting = np.load(output_fitness_file)
       fitness_values = ting['fitness']
    '''
    output_fitness_file = os.path.join(universe.output_folder, "gen%04d_fitness.npz" % universe.generation)
    np.savez(output_fitness_file, fitness=universe.fitness_scores)
    logging.debug("saved scores for generation %i" % universe.generation)


def save_population(universe):
    '''
    save each individual_material as a pickle file named with it's id

    here is how to open a pickled file:
        from codes.genetic_material import IndividualMaterial
        with open(indiv_file, "rb") as f:
            indiv = pkl.load(f)
    '''
    logging.debug("saved each individual from population for generation %i" % universe.generation)
    for indiv in universe.population.population:
        indiv_file = os.path.join(universe.output_folder, "gen_%04d_indiv_%s.pkl" % (universe.generation, indiv.id))
        with open(indiv_file, "wb") as f:
            pkl.dump(indiv, f)