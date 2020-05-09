'''
root/code/population.py

Overview:
factory fills the population with a list of individuals

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages

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