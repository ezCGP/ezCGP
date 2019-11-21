# RUN_UNIVERSE parallely


import numpy as np
from copy import deepcopy
import time

import selections
import gc
import os, sys
import threading
import logging

from individual import Individual, build_individual
from mpi4py import MPI
from mating import Mate


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def split_pop(pop, num_cpu):
    """
    :param pop:
    :param num_cpu:
    :return:
    """
    new_pop = []
    for i in range(num_cpu):
        new_pop.append([])

    len_pop = len(pop)
    for i in range(len_pop):
        indx = i % len(new_pop)
        new_pop[indx].append(pop[i])

    return new_pop

def mate_population(population):
    # mate through population before evaluating and selecting
    logging.info("    MATING")
    # build individuals from genome list (genome_output_values)
    # then (back external) limit the number of total population size to be a multiple of # cpu cores
    # for Jinghua will focus on inside Mating
    # for Sam will focus on converting genome list to list of indivs, and after (end of this function)
    # convert individuals to genome values for multiprocessing

    # convert population of genome lists into individuals
    population = [build_individual(deepcopy(problem.skeleton_genome), deepcopy(genome))
                     for genome in population]

    # initialize mate wrapper
    mate_obj = Mate(population, problem.skeleton_genome)

    # mate and produce two random offspring
    # TODO: extend this to mate any number of offspring (e.g percentage of num cpu core for efficiency)
    mate_list = mate_obj.whole_block_swapping()
    for mate in mate_list:
        if mate.need_evaluate:
            population.append(mate)
        else:
            pass

    # convert population back to genome list for simpler processing
    # population = [population.append(genome_list) for subpop in population for genome_list in subpop]
    population = [ind.get_genome_list() for ind in population]
    return population

def run_universe(population, num_mutants, num_offspring, input_data, labels,
                 block=None):  # be able to select which block we want to evolve or randomly select
    # TODO: Integrate mating into mutation and overall run

    # mutate through the population
    print("    MUTATING")
    for i in range(len(population)):  # don't loop over population and add to population in the loop
        individual = population[i]
        for _ in range(num_mutants):
            mutant = build_individual(problem.skeleton_genome, individual.get_genome_list())
            mutant.mutate()

            if mutant.need_evaluate:
                # then it did for sure mutate
                population.append(mutant)
            else:
                # if mut_prob is < 1 there is a chancce it didn't mutate
                pass
    print("population after mutation", len(population))
    # evaluate the population
    print("    EVALUATING")
    for individual in population:
        if individual.need_evaluate():
            # look up concurrent.futures and queue...
            # maybe make the queue thing a permanent part of universe in evaluating every individual
            # then make solving each node customizable...gt computer nodes, locally on different processes, or on cloud compute service
            # add to queue to evaluate individual
            # evaluate uses multithreading to send individuals to evaluate blocks
            individual.evaluate(problem.x_train, problem.y_train, (problem.x_val, \
                                                                   problem.y_val))
            individual.fitness.values = problem.scoreFunction(actual=problem.y_val, \
                                                              predict=individual.genome_outputs)
            print('Muatated population individual has fitness: {}' \
                  .format(individual.fitness.values))

    # filter population down based off fitness
    # new population done: rank individuals in population and trim down
    print("population after selection ", len(population))
    gc.collect()

    return population  # , eval_queue


if __name__ == '__main__':
    # tracemalloc.start()

    # Init MPI Communication and get CPU rank (ID)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print("Start MPI Universe")

    # set the seed and import scripts
    seed = 5
    np.random.seed(seed)
    # keep these imports after the seed is set for numpy
    import problem

    train_data = problem.x_train
    train_labels = problem.y_train

    final_populations = []  # one for each universe created
    num_universes = 1  # 20
    for i in range(num_universes):
        print("start new run %i" % i)
        start = time.time()

        print("--------------------CREATE UNIVERSE--------------------")
        input_data = train_data
        labels = train_labels
        universe_seed = seed + i
        population_size = 4  # Should be multiple of num of CPUs
        num_mutants, num_offspring = 1, 2

        np.random.seed(universe_seed)

        """
        Each CPU initialize its own subpopulation
        """
        population = []
        for i in range(int(population_size / size)):
            ind = Individual(skeleton=problem.skeleton_genome)
            population.append(ind.get_genome_list())

        for i in range(len(population)):
            print("CPU %i CREATE INDIVIDUAL" % rank)
            individual = build_individual(problem.skeleton_genome,
                                                            population[i])
            print("CPU %i EVALUATE INDIVIDUAL" % rank)
            individual.evaluate(problem.x_train, problem.y_train, (problem.x_val,
                                                                   problem.y_val))  # This probably works. Or it does not. Seems to. Or print statements are broken
            try:

                print("CPU %i Scoring INDIVIDUAL" % rank)
                individual.fitness.values = problem.scoreFunction(actual=problem.y_val,
                                                                  predict=individual.genome_outputs)
                population[i][-1] = individual.fitness.values  # better fix?
                print('CPU {}: Initialized individual has fitness {}'
                      .format(rank, individual.fitness.values))
            except:
                import pdb

                pdb.set_trace()

        """
        Gather initialized population back to Master CPU
        """
        print(population)
        if rank == 0:
            comm.Barrier()
            new_pop = comm.gather(population, root=0)
            print(new_pop)
            print("--------------------END CREATE UNIVERSE--------------------")
            # converting 2D initial genome output list pop into 1D for first mating
            population = []
            for subpop in new_pop:
                for genome_list in subpop:
                    population.append(genome_list)

        generation = -1
        converged = False
        GENERATION_LIMIT = problem.generation_limit  # 199
        SCORE_MIN = problem.score_min  # 1e-1
        start_time = time.time()
        newpath = r'outputs_cifar/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        file_generation = 'outputs_cifar/generation_number.npy'

        """
        Parallel GP ezCGP starts here
        """
        while (not converged) & (generation <= GENERATION_LIMIT):
            """
            Scatter population across all slave cpu
            Mate (1D)
            Scatter (2D)
            Gather (Merge into 1D)
            """
            # 1D population (genome list) array after each iteration of the loop
            # mate individuals and insert into next population
            population = mate_population(population) # needs to be 1D to mate

            # split pop
            population = split_pop(population, size) # becomes 2D

            comm.Barrier()
            scatter_start = time.time()
            population = comm.scatter(population, root=0)
            scatter_end = time.time()

            print("Rank: {} Length: {}".format(rank, len(population)))
            print("-------------RAN UNIVERSE FOR GENERATION: {}-----------".format(generation + 1))
            indPopulation = [build_individual(deepcopy(problem.skeleton_genome), deepcopy(genome))
                             for genome in population]
            run_start = time.time()
            indPopulation = run_universe(indPopulation, num_mutants, num_offspring, input_data, labels)
            run_end = time.time()
            population = [ind.get_genome_list() for ind in indPopulation]

            """
            Gather genomes lists back to Master CPU
            """
            comm.Barrier()
            gather_start = time.time()
            population = comm.gather(population, root=0)
            gather_end = time.time()

            generation_list = []
            converged_list = []

            """
            Master CPU job is
                - Generate invidividuals from the gathered Genome List
                - to select the best candidates for the next generation
            """
            comm.Barrier()
            select_start = time.time()
            if rank == 0:
                generation += 1
                new_pop = []
                for subpop in population:
                    for genome_list in subpop:
                        new_pop.append(genome_list)
                indPopulation = [build_individual(problem.skeleton_genome, genome_list)
                                 for genome_list in new_pop]
                indPopulation, _ = selections.selNSGA2(indPopulation, k=population_size, nd='standard')
                population = [ind.get_genome_list() for ind in indPopulation] # 1D gathering after best selection
                scores = []
                # print(population[0])
                for genome_list in population:
                    scores.append(genome_list[-1])

                if np.min(scores) < SCORE_MIN:
                    converged = True
                else:
                    pass

                # file_pop = 'outputs_cifar/gen%i_pop.npy' % (generation)
                # np.save(file_pop, population)
                # np.save(file_generation, generation)


            select_end = time.time()
            with open("cpu_%i.txt" % rank, "a") as f:
                f.write("Gen %i Scatter: %f, Gather: %f, Run: %f, Select: %f\n" %
                        (
                            generation, (scatter_end - scatter_start), (gather_end - gather_start),
                            (run_end - run_start), (select_end - select_start)
                        )
                        )

            conditions = []
            if rank == 0:
                conditions = [converged, generation]

            comm.Barrier()
            conditions = comm.bcast(conditions, root=0)
            converged = conditions[0]
            generation = conditions[1]

        print("ending universe", time.time() - start_time)
        # ---------------------------------------------------END UNIVERSE-----------------------------------------------------------
        time_complete = time.time() - start
        print("time of generation", time_complete)
        if rank == 0:
            with open("run_time.txt", "a") as f:
                f.write("%f \n" % time_complete)

    # snapshot = tracemalloc.take_snapshot()
    # top = snapshot.statistics('lineno')

    # for stat in top[:20]:
    #     print(stat)
