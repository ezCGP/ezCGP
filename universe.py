### universe.py
# a single instnace of a universe...unique 'state of the world'

# external packages
import numpy as np
from copy import deepcopy
import queue

# my scripts
from individual import Individual


def evaluator_queue():



def run_universe(population, num_mutants, block=None): # be able to select which block we want to evolve or randomly select
    # mate through the population
    mating_list = tournamentSelection(population, k=len(population)) #double check that k works
    for i in range(0,len(mating_list),2):
        parent1 = deepcopy(mating_list[i])
        parent2 = deepcopy(mating_list[i+1])
        offspring_list = parent1.mate(parent2, block)
        if offspring_list is not None:
            for offspring in offspring_list:
                population.append(offspring)
        else:
            # if mate_prob < 1 there is a chance that it didn't mate and return None
            pass

    # mutate through the population
    for i in range(len(population)): # don't loop over population and add to population in the loop
        individual = population[i]
        for _ in range(num_mutants):
            mutant = deepcopy(individual)
            mutant.mutate(block)
            if mutant.need_evaluate:
                # then it did for sure mutate
                population.append(mutant)
            else:
                # if mut_prob is < 1 there is a chancce it didn't mutate
                pass

    # evaluate the population
    for individual in population:
        if individual.need_evaluate:
            # look up concurrent.futures and queue...
            #maybe make the queue thing a permanent part of universe in evaluating every individual
            #then make solving each node customizable...gt computer nodes, locally on different processes, or on cloud compute service
            #
            # add to queue to evaluate individual
            # evaluate uses multithreading to send individuals to evaluate blocks
            eval_queue.put(individual)
            #individual.evaluate(input_data, block)
            #individual.score_fitness(labels)

    return population, eval_queue


def create_universe(input_data, labels, population_size=100, seed=9, num_mutants=4):
    np.random.seed(seed)

    # initialize the population
    population = []
    for i in range(population_size):
        individual = Individual()
        individual.evaluate(genome_inputs=input_data)
        individual.score_fitness(labels=labels)
        population.append(individual)

    eval_queue = queue.Queue()
    generation = 0
    converged = False
    while (not converged) & (generation<=GENERATION_LIMIT):
        generation += 1
        population, eval_queue = run_universe(population, eval_queue num_mutants, num_offpsring)
        # population multiobjective ranking here or right before it get's returned?

        scores = []
        for individual in population:
            score.append(population.fintess)
        if max(scores) > SCORE_LIMIT:
            converged = True
        else:
            pass
