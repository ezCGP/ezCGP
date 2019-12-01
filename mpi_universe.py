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
import problem


def blockPrint():
    """Disable"""
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    """Restore"""
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

    for i in range(len(pop)):
        new_pop[i % len(new_pop)].append(pop[i])

    return new_pop


def merge_pop(pop):
    new_pop = []
    for subpop in pop:
        for genome_list in subpop:
            new_pop.append(genome_list)
    return new_pop


def mate_population(population):
    """
    Mate through population before evaluating and selecting
    Build individuals from genome list (genome_output_values)
    then (back external) limit the number of total population size to be a multiple of # cpu cores
    for Jinghua will focus on inside Mating
    for Sam will focus on converting genome list to list of indivs, and after (end of this function)
    convert individuals to genome values for multiprocessing
    convert population of genome lists into individuals
    :param population:
    :return:
    """
    print("    MATING")
    population = [build_individual(problem.skeleton_genome, genome)
                  for genome in population]

    #for i in range(problem.N_OFFSPRING):
    # Initialize mate wrapper
    mate_obj = Mate(population, problem.skeleton_genome)
    
    # Mate and produce two random offspring
    # TODO: extend this to mate any number of offspring (e.g percentage of num cpu core for efficiency)
    for i in range(problem.N_OFFSPRING):
        mate_list = mate_obj.whole_block_swapping() # creates two offspring
        for mate in mate_list:
            if mate.need_evaluate:
                population.append(mate)
            else:
                pass

    return [ind.get_genome_list() for ind in population]


def run_universe(population, num_mutants, num_offspring, input_data, labels, block=None):
    """
    # Be able to select which block we want to evolve or randomly select
    # TODO: Integrate mating into mutation and overall run
    :param population:
    :param num_mutants:
    :param num_offspring:
    :param input_data:
    :param labels:
    :param block:
    :return:
    """
    print("    MUTATING")
    for i in range(len(population)):
        individual = population[i].get_genome_list()
        for _ in range(num_mutants):
            mutant = build_individual(problem.skeleton_genome, individual)
            mutant.mutate()

            # Check if mutant needs to be evaluated
            # If mut_prob is < 1 there is a chance it didn't mutate
            if mutant.need_evaluate:
                population.append(mutant)
            else:
                pass

    print("Population after mutation", len(population))

    print("    EVALUATING")
    for individual in population:
        if individual.need_evaluate():
            """
            Look up concurrent.futures and queue...
            Maybe make the queue thing a permanent part of universe in evaluating every individual
            then make solving each node customizable...gt computer nodes, locally on different processes,
            or on cloud compute service
            Add to queue to evaluate individual
            Evaluate uses multithreading to send individuals to evaluate blocks
            """
            individual.evaluate(problem.x_train, problem.y_train, (problem.x_val, problem.y_val))
            individual.fitness.values = problem.scoreFunction(actual=problem.y_val,
                                                              predict=individual.genome_outputs)

            print('Mutated population individual has fitness: {}'.format(individual.fitness.values))
            print('Genome shape:') # replace with actual visualization code later
            for i in range(1,individual.num_blocks+1):
                curr_block = individual.skeleton[i]["block_object"]
                for active_node in curr_block.active_nodes:
                    print(curr_block[active_node])


    gc.collect()

    return population  # , eval_queue


if __name__ == '__main__':

    # Init MPI Communication and get CPU rank (ID)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print("Start MPI Universe")

    seed = problem.SEED
    np.random.seed(seed)
    # keep these imports after the seed is set for numpy

    train_data = problem.x_train
    train_labels = problem.y_train

    final_populations = []  # one for each universe created
    for i in range(problem.N_UNIVERSE):
        print("===== STARTING UNIVERSE: ", i)
        start = time.time()

        print("--------------------CREATE UNIVERSE--------------------")
        input_data = train_data
        labels = train_labels
        population_size = problem.POP_SIZE  # Should be multiple of num of CPUs

        np.random.seed(seed + i)

        """
        Each CPU initialize its own subpopulation
        """
        population = []
        for i in range(int(population_size / size)):
            ind = Individual(skeleton=problem.skeleton_genome)
            population.append(ind.get_genome_list())

        for i in range(len(population)):
            print("CPU %i CREATE INDIVIDUAL" % rank)
            individual = build_individual(problem.skeleton_genome, population[i])
            print("CPU %i EVALUATE INDIVIDUAL" % rank)
            individual.evaluate(problem.x_train, problem.y_train, (problem.x_val, problem.y_val))
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
        population = comm.gather(population, root=0)
        if rank == 0:
            # Converting 2D initial genome output list pop into 1D for first mating
            population = merge_pop(population)

        generation = 0
        converged = False
        start_time = time.time()
        newpath = r'outputs_cifar/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        file_generation = 'outputs_cifar/generation_number.npy'

        """
        Parallel GP ezCGP starts here
        """
        while (not converged) & (generation < problem.GEN_LIMIT):
            """
            Scatter population across all slave cpu
            Mate (1D)
            Scatter (2D)
            Gather (Merge into 1D)

            1D population (genome list) array after each iteration of the loop
            mate individuals and insert into next population
            """

            if rank == 0:
                if len(population) == 0:
                    population = mate_population(population)  # needs to be 1D to mate
                    population = split_pop(population, size)  # becomes 2D


            comm.Barrier()
            scatter_start = time.time()
            population = comm.scatter(population, root=0)
            scatter_end = time.time()

            print("Rank: {} Length: {}".format(rank, len(population)))
            print("-------------RAN UNIVERSE FOR GENERATION: {}-----------".format(generation + 1))
            indPopulation = [build_individual(problem.skeleton_genome, genome) for genome in population]
            run_start = time.time()
            indPopulation = run_universe(indPopulation, problem.N_MUTANTS, problem.N_OFFSPRING, input_data, labels)
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
                new_pop = merge_pop(population)
                indPopulation = [build_individual(problem.skeleton_genome, genome_list)
                                 for genome_list in new_pop]
                """
                Filter population down based off fitness
                New population done: rank individuals in population and trim down
                """
                indPopulation, _ = selections.selNSGA2(indPopulation, k=population_size, nd='standard')
                population = [ind.get_genome_list() for ind in indPopulation]
                scores = []
                for genome_list in population:
                    scores.append(genome_list[-1])

                if np.min(scores) < problem.MIN_SCORE:
                    converged = True
                else:
                    pass

                file_pop = 'outputs_cifar/gen%i_pop.npy' % (generation)
                np.save(file_pop, population)
                np.save(file_generation, generation)

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

        print("===== ENDING UNIVERSE ", i, "total time: ", time.time() - start_time)


        time_complete = time.time() - start
        print("Time of generation", time_complete)
        if rank == 0:
            with open("run_time.txt", "a") as f:
                f.write("%f \n" % time_complete)
