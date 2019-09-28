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
    pop_length = len(pop)
    new_pop = []
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

        # -------------------------------CREATE UNIVERSE---------------------------------------
        print("--------------------CREATE UNIVERSE--------------------")
        input_data = train_data
        labels = train_labels
        universe_seed = seed + i
        population_size = 2
        num_mutants, num_offpsring = 4, 2

        np.random.seed(universe_seed)

        if rank == 0:
            population = []
            for i in range(population_size):
                ind = Individual(skeleton=problem.skeleton_genome)
                ind.clear_rec()  # clear rec to allow deepcopy to work
                ind = deepcopy(ind)  # need to deepcopy individual so that the dataset does not cause pickling error
                population.append(ind)
            population = split_pop(population, size)
            print("Pop length", len(population))
            for p in population:
                print("Sub pop", len(p))
        else:
            population = None

        population = comm.scatter(population, root=0)

        print("CReATE UNIVERSE SCATTER POP")
        print(population)
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
            generation += 1
            population = run_universe(population, num_mutants, num_offpsring, input_data, labels)
            scores = []
            for individual in population:
                scores.append(individual.fitness.values[0])
            print("-------------RAN UNIVERSE FOR GENERATION: {}-----------".format(generation + 1))
            print(generation, np.min(scores))
            if np.min(scores) < SCORE_MIN:
                converged = True
            else:
                pass
            if (generation % 10 == 0) or (converged):
                # plot
                # import pdb; pdb.set_trace()
                sample_best = population[np.random.choice(a=np.where(np.min(scores) == scores)[0], size=1)[0]]
                try:
                    print(sample_best.genome_outputs[0])
                except:
                    import pdb

                    pdb.set_trace()
                '''
                #sample_best = population[np.where(np.min(scores)==scores)[0][0]]
                #print(problem.x_train)
                #print(sample_best.genome_outputs)
                plt.figure()
                plt.plot(problem.x_train[0], problem.y_train, '.')
                #testY = solutions[run].testEvaluate()
                plt.plot(problem.x_train[0], sample_best.genome_outputs[0], '.')
                #plt.legend(['Weibull','Test Model Fit'])
                plt.legend(['log(x)','Test Model Fit'])
                #plt.show()
                Path('outputs').mkdir(parents=True, exist_ok=True) #should help work on all OS
                filepath = 'outputs/seed%i_gen%i.png' % (universe_seed, generation)
                plt.savefig(filepath)
                plt.close()
                '''
            # file_pop = 'outputs_cifar/gen%i_pop.npy' % (generation)
            # np.save(file_pop, population)
            # np.save(file_generation, generation)

            converged_list = None
            generation_list = None

            if rank == 0:
                converged_list = [converged for i in range(size)]
                generation_list = [generation for i in range(size)]

            print("Converged: ", converged)
            print("Generation: ", generation)

            print("Length: ", sys.getsizeof(population[0]))
            # if rank != 0:
            #     comm.send(population[0], dest=0, tag=11)
            # if rank == 0:
            #     for i in range(size):
            #         population = comm.recv(i, tag=11)
            population = comm.gather(population, root=0)

            if rank == 0:
                population, _ = selections.selNSGA2(population, k=population_size, nd='standard')

            converged = comm.scatter(converged_list, root=0)
            generation = comm.scatter(generation_list, root=0)

        print("ending universe", time.time() - start_time)
        # ---------------------------------------------------END UNIVERSE-----------------------------------------------------------

        print("time of generation", time.time() - start)
