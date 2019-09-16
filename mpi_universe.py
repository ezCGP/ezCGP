# RUN_UNIVERSE parallely


import numpy as np
from copy import deepcopy
import time
from individual import Individual
import problem
import selections
import gc
import os
from mpi4py import MPI


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
    population, _ = selections.selNSGA2(population, k=pop_size, nd='standard')
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
        input_data = train_data
        labels = train_labels
        universe_seed = seed + i
        population_size = 500
        num_mutants, num_offpsring = 4, 2

        np.random.seed(universe_seed)

        """
        Population needs to contains num of Individual such that it divides num of CPU
        Example:
        3 CPU -> populations can contain 3, 6, or 9,.. number of elements
        Each element can be a sub-population that each CPU will perform computation on
        """
        print("--------------------Scattering Population Start--------------------")
        population = []
        if rank == 0:
            buffer = []
            for i in range(population_size):
                individual = Individual(skeleton=problem.skeleton_genome)
                individual.evaluate(problem.x_train, problem.y_train, (problem.x_val, \
                                                                       problem.y_val))
                # individual.score_fitness(labels=labels)
                try:
                    individual.fitness.values = problem.scoreFunction(actual=problem.y_val, \
                                                                      predict=individual.genome_outputs)
                    print('Initialized individual has fitness: {}' \
                          .format(individual.fitness.values))
                except:
                    import pdb

                    pdb.set_trace()
                buffer.append(individual)
                del individual

            population = split_pop(buffer, size)

        """
        Scatter population across all slave CPU
        """
        population = comm.scatter(population, root=0)

        print("Rank: {} Length: {}".format(rank, len(population)))
        print("--------------------Scattering Population End--------------------")

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
            Scatter population
            """
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
            file_pop = 'outputs_cifar/gen%i_pop.npy' % (generation)
            np.save(file_pop, population)
            np.save(file_generation, generation)

        print("ending universe", time.time() - start_time)
        # ---------------------------------------------------END UNIVERSE-----------------------------------------------------------

        print("time of generation", time.time() - start)
