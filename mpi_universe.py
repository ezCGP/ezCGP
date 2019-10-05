# RUN_UNIVERSE parallely


import numpy as np
from copy import deepcopy
import time
from individual import Individual
import selections
import gc
import os, sys
from mpi4py import MPI
import logging


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def create_universe(labels, pop_sz=20, seed=9, mutatn_sz=4, offspring_sz=2, comm=None, size=None, rank=None):
    np.random.seed(seed)

    if rank == 0:
        population = [Individual(skeleton=problem.skeleton_genome) for i in range(pop_sz)]
        population = split_pop(population, size)
    else:
        population = None

    population = comm.scatter(population, root=0)

    print("CReATE UNIVERSE SCATTER POP")
    # parallelize
    for i in range(len(population)):
        individual = population[i]
        individual.evaluate(problem.x_train, problem.y_train, (problem.x_val,
                                                               problem.y_val))
        # individual.score_fitness(labels=labels)
        try:
            individual.fitness.values = problem.scoreFunction(actual=problem.y_val,
                                                              predict=individual.genome_outputs)
            print('Initialized individual has fitness: {}'
                  .format(individual.fitness.values))
        except:
            import pdb
            pdb.set_trace()

    population = comm.gather(population, root=0)
    return population


def split_pop(pop, num_cpu):
    """
    :param pop:
    :param num_cpu:
    :return:
    """
    # num_cpu -= 1
    pop_length = len(pop)
    new_pop = []
    # new_pop.append([])
    bucket_size = int(pop_length / num_cpu)
    left_over = []
    for i in range(0, pop_length, bucket_size):
        end = i + bucket_size
        if end <= pop_length:
            new_pop.append(
                pop[i: end]
            )
        else:
            left_over = pop[i:]

    if len(left_over) > 0:
        for i in range(len(left_over)):
            new_pop[i].append(
                left_over[i]
            )
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
            mutant = deepcopy(
                individual)  # can cause recursive copying issues with tensor blocks, so we empty them in blocks.py evaluate() bottom
            print("deepcopied")
            mutant.mutate(block)
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


    # Read in Data
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
        population_size = 3
        num_mutants, num_offspring = 4, 2

        np.random.seed(universe_seed)

        if rank == 0:
            population = []
            for i in range(population_size):
                print("Indivilual ", i)
                ind = Individual(skeleton=problem.skeleton_genome)
                ind.clear_rec()  # clear rec to allow deepcopy to work
                ind = deepcopy(ind)  # need to deepcopy individual so that the dataset does not cause pickling error
                population.append(ind)

            population = split_pop(population, size)
            print("Pop length", len(population))
            print("Len pop: ", len(population))
            for p in population:
                print("Sub pop", len(p))
        else:
            population = None
        population = comm.scatter(population, root=0)

        print("CREATE UNIVERSE SCATTER POP")

        """
        START PARALLEL INDIVIDUAL EVALUATE
        """

        for i in range(len(population)):
            individual = population[i]
            individual.evaluate(problem.x_train, problem.y_train, (problem.x_val,
                                                                   problem.y_val))
            # individual.score_fitness(labels=labels)
            try:
                individual.fitness.values = problem.scoreFunction(actual=problem.y_val,
                                                                  predict=individual.genome_outputs)
                print('Initialized individual has fitness: {}'
                      .format(individual.fitness.values))
            except:
                import pdb

                pdb.set_trace()

        """
        END PARALLEL INDIVIDUAL EVALUATE
        """
        print("Length: ", len(population))
        for i in population:
            i.clear_rec()
        population = comm.gather(population, root=0)

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

            print("Len before run universe: ", len(population))
            population = run_universe(population, num_mutants, num_offspring, input_data, labels)
            print("Len after run universe: ", len(population))

            """
            Gather scores
            """

            for ind in population:
                ind.clear_rec()  # clear rec to allow deepcopy to work
            population = comm.gather(population, root=0)

            generation_list = []
            converged_list = []
            if rank == 0:
                generation += 1
                new_pop = []
                for subpop in population:
                    for individual in subpop:
                        new_pop.append(individual)

                population, _ = selections.selNSGA2(new_pop, k=population_size, nd='standard')
                scores = []

                for individual in population:
                    scores.append(individual.fitness.values[0])

                if np.min(scores) < SCORE_MIN:
                    converged = True
                else:
                    pass
                if (generation % 10 == 0) or converged:
                    # plot
                    # import pdb; pdb.set_trace()
                    sample_best = population[np.random.choice(a=np.where(np.min(scores) == scores)[0], size=1)[0]]
                    try:
                        print(sample_best.genome_outputs[0])
                    except:
                        import pdb
                        pdb.set_trace()
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
