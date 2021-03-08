'''
root/code/population.py

Overview:
factory fills the population with a list of individuals

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
import itertools

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from codes.utilities.custom_logging import ezLogging



class PopulationDefinition():
    '''
    words
    '''
    def __init__(self,
                population_size):
        #self.pop_size = population_size #moved to problem.pop_size
        self.population = []


    def __getitem__(self, indiv_index):
        '''
        TODO
        '''
        return self.population[indiv_index]


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
        say we have 27 individuals and we want 7 subpops

        start by assigning equal number to the 7 groups with 27//7 which is 3
            [3,3,3,3,3,3,3]
        then go through the remainder and add +1

        # then split up the population by those sizes
        '''
        subpop_sizes = [len(self.population)//num_sub_pops] * num_sub_pops
        for ith_pop in range(len(self.population)%num_sub_pops):
            subpop_sizes[ith_pop] += 1

        subpops = []
        position = 0
        for size in subpop_sizes:
            subpops.append(self.population[position:position+size])
            position += size

        self.population = subpops


    def merge_subpopulations(self, subpops):
        '''
        if we had a list of list of individual_materials in subpops,
        then we'd want to append them into a single large list and 
        assign to self.population
        '''
        self.population = list(itertools.chain.from_iterable(subpops))
        ezLogging.info("Combined %i sub populations into a single population" % (len(subpops)))