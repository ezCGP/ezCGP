### universe.py
# a single instnace of a universe...unique 'state of the world'

# external packages
import numpy as np
from copy import deepcopy
import queue
import matplotlib.pyplot as plt
import time
from pathlib import Path

# my scripts
from individual import Individual
import problem
import selections


def evaluator_queue():
    pass


def run_universe(population, num_mutants, num_offspring, input_data, labels, block=None): # be able to select which block we want to evolve or randomly select
    # mate through the population
    pop_size = len(population)
    '''
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
    '''
    # mutate through the population
    print("    MUTATING")
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
    print("    EVALUATING")
    for individual in population:
        if individual.need_evaluate():
            # look up concurrent.futures and queue...
            #maybe make the queue thing a permanent part of universe in evaluating every individual
            #then make solving each node customizable...gt computer nodes, locally on different processes, or on cloud compute service
            #
            # add to queue to evaluate individual
            # evaluate uses multithreading to send individuals to evaluate blocks
            #eval_queue.put(individual)
            individual.evaluate(input_data)
            #individual.score_fitness(labels)
            individual.fitness.values = problem.scoreFunction(actual=labels, predict=individual.genome_outputs)

    # filter population down based off fitness
    # new population done: rank individuals in population and trim down
    #print("prep for next pop")
    population, _ = selections.selNSGA2(population, k=pop_size, nd='standard')

    return population #, eval_queue


def create_universe(input_data, labels, population_size=100, universe_seed=9, num_mutants=4, num_offpsring=2):
    np.random.seed(universe_seed)

    # initialize the population
    population = []
    for i in range(population_size):
        individual = Individual(skeleton=problem.skeleton_genome)
        individual.evaluate(data=input_data)
        #individual.score_fitness(labels=labels)
        individual.fitness.values = problem.scoreFunction(actual=labels, predict=individual.genome_outputs)
        population.append(individual)
        del individual

    #eval_queue = queue.Queue()
    generation = -1
    converged = False
    GENERATION_LIMIT = problem.generation_limit #199
    SCORE_MIN = problem.score_min #1e-1
    start_time = time.time()
    while (not converged) & (generation<=GENERATION_LIMIT):
        generation += 1
        #population, eval_queue = run_universe(population, eval_queue num_mutants, num_offpsring)
        population = run_universe(population, num_mutants, num_offpsring, input_data, labels)
        # population multiobjective ranking here or right before it get's returned?

        scores = []
        for individual in population:
            scores.append(individual.fitness.values[0])
        print(generation, np.min(scores))
        if np.min(scores) < SCORE_MIN:
            converged = True
        else:
            pass
        if (generation%10 == 0) or (converged):
            # plot
            #import pdb; pdb.set_trace()
            sample_best = population[np.random.choice(a=np.where(np.min(scores)==scores)[0], size=1)[0]]
            #sample_best = population[np.where(np.min(scores)==scores)[0][0]]
            #print(problem.x_train)
            #print(sample_best.genome_outputs)
            plt.figure()
            plt.plot(problem.x_train[1], problem.y_train, '.')
            #testY = solutions[run].testEvaluate()
            plt.plot(problem.x_train[1], sample_best.genome_outputs[0], '.')
            #plt.legend(['Weibull','Test Model Fit'])
            plt.legend(['log(x)','Test Model Fit'])
            #plt.show()
            Path('outputs').mkdir(parents=True, exist_ok=True) #should help work on all OS
            filepath = 'outputs/seed%i_gen%i.png' % (universe_seed, generation)
            plt.savefig(filepath)
            plt.close()
    print("ending universe", time.time()-start_time)
