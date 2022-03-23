'''
root/code/population.py

Overview:
factory fills the population with a list of individuals

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
import itertools
import deap.tools

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
    def __init__(self):
        #self.pop_size = population_size #moved to problem.pop_size
        self.population = []
        self.hall_of_fame = None


    def __setitem__(self, node_index, value):
        self.population[node_index] = value


    def __getitem__(self, node_index):
        return self.population[node_index]


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


    def setup_hall_of_fame(self, maxsize):
        '''
        https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.HallOfFame

        letting hall_of_fame use to be optional
        '''
        def similarity_equation(a, b):
            '''
            there is an option to pass in "an equivalence operator between two individuals, optional".
            if it is used to make sure it's not adding duplicate individuals then I'll make my own
            based off individual id's

            return if equal. false otherwise.
            assuming a and b are IndividualMaterial objects
            '''
            if a.id == b.id:
                return True
            else:
                return False

        self.hall_of_fame = deap.tools.HallOfFame(maxsize=maxsize,
                                                  similar=similarity_equation)
        ezLogging.debug("Established 'Hall of Fame' with maxsize %i" % maxsize)


    def update_hall_of_fame(self):
        '''
        https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.HallOfFame.update
        '''
        if self.hall_of_fame is not None:
            # filter out dead people
            alive_population = []
            for indiv in self.population:
                if not indiv.dead:
                    alive_population.append(indiv)
            self.hall_of_fame.update(alive_population)
            ezLogging.debug("Updated Hall of Fame to size %i" % (len(self.hall_of_fame.items)))


    def get_pareto_front(self, use_hall_of_fame=False, first_front_only=False):
        '''
        https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.sortNondominated
        https://github.com/DEAP/deap/blob/master/deap/tools/emo.py#L53
        '''
        if use_hall_of_fame:
            individuals = self.hall_of_fame.items()
        else:
            individuals = population.population
        k = len(individuals)
        fronts = deap.tools.sortNondominated(individuals, k, first_front_only)
        ezLogging.debug("Calculated and Found %i Pareto Fronts" % (len(fronts)))
        return fronts
