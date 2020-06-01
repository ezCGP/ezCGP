'''
root/code/universe.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
from typing import List

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems.problem_definition import ProblemDefinition_Abstract
from codes.utilities import selections




class UniverseDefinition():
    '''
    TODO
    '''
    def __init__(self,
                problem: ProblemDefinition_Abstract,
                output_folder: str):
        '''
        electing to keep Problem class separate from Universe class...
        if we ever need a Problem attribute, just pass in the instance of Problem as an arguement
        ...as opposed to having a copy of the Problem instance as an attribute of Universe
        '''
        self.factory = problem.Factory()
        self.population = self.factory.build_population(problem.indiv_def, problem.pop_size)
        self.problem = problem
        self.output_folder = output_folder
        self.converged = False


    def parent_selection(self):
        '''
        TODO
        '''
        return selections.selTournamentDCD(self.population.population, k=len(self.population.population))


    def evolve_population(self, problem: ProblemDefinition_Abstract):
        '''
        TODO
        '''
        # MATE
        children = []
        mating_list = self.parent_selection()
        for ith_indiv in range(0, len(mating_list), 2):
            parent1 = mating_list[ith_indiv]
            parent2 = mating_list[ith_indiv+1]
            children += problem.indiv_def.mate(parent1, parent2)
        self.population.add_next_generation(children)

        # MUTATE
        mutants = []
        for individual in self.population.population:
            mutants += problem.indiv_def.mutate(individual)
        self.population.add_next_generation(mutants)


    def evaluate_score_population(self, problem: ProblemDefinition_Abstract):
        '''
        TODO
        '''
        self.fitness_scores = []
        for indiv in self.population.population:
            # EVALUATE
            problem.indiv_def.evaluate(indiv, problem.data.x_train)
            # SCORE
            problem.objective_functions(indiv)
            self.fitness_scores.append(indiv.fitness.values)


    def population_selection(self):
        '''
        TODO
        '''
        self.population.population, _ = selections.selNSGA2(self.population.population, self.population.pop_size, nd='standard')


    def check_convergence(self, problem_def: ProblemDefinition_Abstract):
        '''
        Should update self.converged
        '''
        problem_def.check_convergence(self)



    def run(self, problem: ProblemDefinition_Abstract):
        '''
        assumes a population has only been created and not evaluatedscored
        '''
        self.generation = 0
        self.evaluate_score_population(problem)
        self.population_selection()
        while not self.converged:
            self.generation += 1
            self.evolve_population(problem)
            self.evaluate_score_population(problem)
            self.population_selection()
            self.check_convergence(problem)



class MPIUniverseDefinition(UniverseDefinition):
    pass