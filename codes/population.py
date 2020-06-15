'''
root/code/population.py

Overview:
factory fills the population with a list of individuals

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
import logging
import itertools

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root



class PopulationDefinition():
    '''
    words
    '''
    def __init__(self,
                population_size):
        self.pop_size = population_size
        self.population = []


    def get_fitness(self):
        '''
        TODO
        '''
        fitness = []
        for indiv in self.population:
            fitness.append(indiv.fitness.value)
        return fitness


    def add_next_generation(self, next_generation):
        '''
        not clear if we actually need this...here just incase if we find it better to handle adding individuals
        differently that just immediately adding to the rest of the population

        assuming next_generation is a list
        '''
        self.population += next_generation


    def split_population(self, num_sub_pops):
        '''
        TODO
        '''
        pass


    def merge_subpopulations(self, subpops):
        '''
        if we had a list of list of individual_materials in subpops,
        then we'd want to append them into a single large list and 
        assign to self.population
        '''
        self.population = list(itertools.chain.from_iterable(subpops))
        logging.info("Combined %i sub populations into a single population" % (len(subpops)))