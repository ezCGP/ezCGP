# RUN_UNIVERSE parallely


import numpy as np
from copy import deepcopy
import time
from individual import Individual, create_individual_from_genome_list
import selections
import gc
import os, sys
from mpi4py import MPI
import logging
import tracemalloc


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


def run_universe(population, num_mutants, num_offspring, input_data, labels,
                 block=None):  # be able to select which block we want to evolve or randomly select
    # mate through the population
    pop_size = len(population)
    # mutate through the population
    print("    MUTATING")
    for i in range(len(population)):  # don't loop over population and add to population in the loop
        individual = population[i]
        for _ in range(num_mutants):
            mutant = create_individual_from_genome_list(problem.skeleton_genome, individual.get_genome_list())

            # mutant = deepcopy(
            #     individual)  # can cause recursive copying issues with tensor blocks, so we empty them in blocks.py evaluate() bottom
            print("deepcopied")
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
    tracemalloc.start()

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
    import universe

    # data_shared = comm.bcast(data_shared, root=0)

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
        population_size = 4
        num_mutants, num_offspring = 1, 2

        np.random.seed(universe_seed)

        if rank == 0:
            population = []
            for i in range(population_size):
                print("Indivilual ", i)
                ind = Individual(skeleton=problem.skeleton_genome)
                # ind.clear_rec()  # clear rec to allow deepcopy to work
                # ind = deepcopy(ind)  # probably no need to deepcopy
                population.append(ind.get_genome_list())

            population = split_pop(population, size)
            print("Pop length", len(population))
            print("Len pop: ", len(population))
            for p in population:
                print("Sub pop", len(p))
        else:
            population = None
        start = time.time()
        population = comm.scatter(population, root=0)

        print("CREATE UNIVERSE SCATTER POP")

        """
        START PARALLEL INDIVIDUAL EVALUATE
        """

        for i in range(len(population)):
            individual = create_individual_from_genome_list(problem.skeleton_genome,
                                                            population[i])
            individual.evaluate(problem.x_train, problem.y_train, (problem.x_val,
                                                                   problem.y_val))  # This probably works. Or it does not. Seems to. Or print statements are broken
            try:

                individual.fitness.values = problem.scoreFunction(actual=problem.y_val,
                                                                  predict=individual.genome_outputs)
                population[i][-1] = individual.fitness.values  # better fix?
                print('Initialized individual has fitness: {}'
                      .format(individual.fitness.values))
            except:
                import pdb

                pdb.set_trace()

        """
        END PARALLEL INDIVIDUAL EVALUATE
        """
        print("Length: ", len(population))
        population = comm.gather(population, root=0)
        print("Total Time", time.time() - start)
        print("--------------------END CREATE UNIVERSE--------------------")

        # eval_queue = queue.Queue()
        generation = -1
        converged = False
        GENERATION_LIMIT = problem.generation_limit  # 199
        SCORE_MIN = problem.score_min  # 1e-1
        start_time = time.time()
        newpath = r'outputs_cifar/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        file_generation = 'outputs_cifar/generation_number.npy'

        while (not converged) & (generation <= GENERATION_LIMIT):
            """
            SCATTER POPULATION ACROSS ALL SLAVE CPU
            """
            population = comm.scatter(population, root=0)
            print("Rank: {} Length: {}".format(rank, len(population)))
            print("--------------------Scattering Population End--------------------")
            print("-------------RAN UNIVERSE FOR GENERATION: {}-----------".format(generation + 1))
            print("Genome 1", population[0])
            indPopulation = [create_individual_from_genome_list(deepcopy(problem.skeleton_genome), deepcopy(genome))
                             for genome in population]
            indPopulation = run_universe(indPopulation, num_mutants, num_offspring, input_data, labels)
            population = [ind.get_genome_list() for ind in indPopulation]

            """
            Gather scores
            """

            # for ind in population:
            #     ind.clear_rec()  # clear rec to allow deepcopy to work
            population = comm.gather(population, root=0)

            generation_list = []
            converged_list = []
            if rank == 0:
                generation += 1
                new_pop = []
                for subpop in population:
                    for genome_list in subpop:
                        new_pop.append(genome_list)
                indPopulation = [create_individual_from_genome_list(problem.skeleton_genome, genome_list)
                                 for genome_list in new_pop]
                indPopulation, _ = selections.selNSGA2(indPopulation, k=population_size, nd='standard')
                population = [ind.get_genome_list() for ind in indPopulation]
                scores = []
                print(population[0])
                for genome_list in population:
                    scores.append(genome_list[-1])

                if np.min(scores) < SCORE_MIN:
                    converged = True
                else:
                    pass
                # if (generation % 10 == 0) or converged:
                #     # plot
                #     # import pdb; pdb.set_trace()
                #     sample_best = population[np.random.choice(a=np.where(np.min(scores) == scores)[0], size=1)[0]]
                #     try:
                #         print(sample_best.genome_outputs[0])
                #     except:
                #         import pdb
                #         pdb.set_trace()
                print("Len pop before split: ", len(population))
                # pdb.set_trace()

                file_pop = 'outputs_cifar/gen%i_pop.npy' % (generation)
                np.save(file_pop, population)
                np.save(file_generation, generation)

                population = split_pop(population, size)

            # print("Converged: ", converged)
            # print("Generation: ", generation)
            # print("Length: ", sys.getsizeof(population[0]))

            converged = comm.bcast(converged, root=0)
            generation = comm.bcast(generation, root=0)

        print("ending universe", time.time() - start_time)
        # ---------------------------------------------------END UNIVERSE-----------------------------------------------------------
        print("time of generation", time.time() - start)

    snapshot = tracemalloc.take_snapshot()
    top = snapshot.statistics('lineno')

    for stat in top[:20]:
        print(stat)
