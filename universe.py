# External packages
import numpy as np
from copy import deepcopy
import time
import gc
import os
import queue
import matplotlib.pyplot as plt
from pathlib import Path

# my scripts
from individual import Individual
import logging
import problem
import selections
from mating import Mate
import visualize

def evaluator_queue():
    pass


def run_universe(population,
                 num_mutants,
                 num_offspring,
                 input_data,
                 labels,
                 block=None):

    pop_size = len(population)
    '''
    logging.info("    MATING")
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


    logging.info("    MATING")
    mate_obj = Mate(population, problem.skeleton_genome)
    mate_list = mate_obj.whole_block_swapping()
    for mate in mate_list:
        if mate.need_evaluate:
            population.append(mate)
        else:
            pass

    # logging.info("    MUTATING")
    # for i in range(len(population)):
    #     individual = population[i]
    #     for _ in range(num_mutants):
    #         '''
    #         Can cause recursive copying issues with tensor blocks,
    #         so we empty them in blocks.py evaluate() bottom
    #         '''
    #         mutant = deepcopy(individual)
    #         mutant.mutate(block)
    #         if mutant.need_evaluate:
    #             population.append(mutant)
    #         else:
    #             # if mut_prob is < 1 there is a chance it didn't mutate
    #             pass

    logging.info("    EVALUATING")
    for individual in population:
        if individual.need_evaluate():
            '''
            Look up concurrent.futures and queue...
            Maybe make the queue thing a permanent part of universe in evaluating every individual
            then make solving each node customizable...gt computer nodes,
            locally on different processes, or on cloud compute service
            add to queue to evaluate individual
            evaluate uses multithreading to send individuals to evaluate blocks
            '''
            individual.evaluate(problem.x_train,
                                problem.y_train,
                                (problem.x_val, problem.y_val))
            individual.fitness.values = problem.scoreFunction(actual=problem.y_val,
                                                              predict=individual.genome_outputs)
            logging.info('Mutated population individual has fitness: {}'
                         .format(individual.fitness.values))

    '''
    Filter population down based off fitness
    new population done: rank individuals in population and trim down
    '''
    population, _ = selections.selNSGA2(population, k=pop_size, nd='standard')
    logging.info("Population after selection " + str(len(population)))
    gc.collect()

    return population


def create_universe(input_data,
                    labels,
                    population_size=8,
                    universe_seed=9,
                    num_mutants=4,
                    num_offpsring=2):
    np.random.seed(universe_seed)

    '''Initialize the population'''
    population = []
    for i in range(population_size):
        individual = Individual(skeleton=problem.skeleton_genome)
        individual.evaluate(problem.x_train, problem.y_train, (problem.x_val, problem.y_val))

        try:
            individual.fitness.values = problem.scoreFunction(actual=problem.y_val,
                                                              predict=individual.genome_outputs)
            logging.info('Initialized individual has fitness: {}'.format(individual.fitness.values))

        except:
            import pdb
            pdb.set_trace()

        population.append(individual)
        del individual
        gc.collect()

    generation = -1
    converged = False
    GENERATION_LIMIT = problem.generation_limit
    SCORE_MIN = problem.score_min
    start_time = time.time()
    newpath = r'outputs_cifar_augment/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    file_generation = 'outputs_cifar_augment/generation_number.npy'

    while (not converged) & (generation <= GENERATION_LIMIT):
        generation += 1
        population = run_universe(population, num_mutants, num_offpsring, input_data, labels)
        scores = []
        list_best_sample = []

        for individual in population:
            scores.append(individual.fitness.values[0])

        logging.info("-------------RAN UNIVERSE FOR GENERATION: {}-----------".format(generation + 1))
        logging.info(str(generation) + str(np.min(scores)))

        if np.min(scores) < SCORE_MIN:
            converged = True
        else:
            pass
        if (generation % 10 == 0) or converged:
            '''At this point, it either finished all generations or it converged'''

            # import pdb
            # pdb.set_trace()

            sample_best = population[np.random.choice(a=np.where(np.min(scores) == scores)[0],
                                                      size=1)[0]]

            list_best_sample.append(vars(sample_best))

            try:
                logging.info("best " + str(sample_best.genome_outputs[0]))
                visualize.Visualizer(sample_best, output_path="csv_individual_visualization/sample_best_{}.csv".format(generation)).create_csv()

            except:
                import pdb
                pdb.set_trace()
            '''
            sample_best = population[np.where(np.min(scores)==scores)[0][0]]
            logging.info(problem.x_train)
            logging.info(sample_best.genome_outputs)
            plt.figure()
            plt.plot(problem.x_train[0], problem.y_train, '.')
            testY = solutions[run].testEvaluate()
            plt.plot(problem.x_train[0], sample_best.genome_outputs[0], '.')
            plt.legend(['Weibull','Test Model Fit'])
            plt.legend(['log(x)','Test Model Fit'])
            plt.show()
            Path('outputs').mkdir(parents=True, exist_ok=True) #should help work on all OS
            filepath = 'outputs/seed%i_gen%i.png' % (universe_seed, generation)
            plt.savefig(filepath)
            plt.close()
            '''

        path = r'outputs_cifar/'
        if not os.path.exists(path):
            os.makedirs(path)
        file_pop = 'outputs_cifar/gen%i_pop.npy' % generation
        np.save(file_pop, population)
        np.save(file_generation, generation)

        #Save best individuals of each generation
        pathV = r'outputs_visualization/'
        if not os.path.exists(pathV):
            os.makedirs(pathV)
        file_best_sample = 'outputs_visualization/gen%i_pop.npy' % generation
        np.save(file_best_sample, list_best_sample)
    logging.info("ending universe" + str(time.time() - start_time))
