import numpy as np
import os
import matplotlib.pyplot as plt
from hypervolume import HyperVolume

def draw_analysis():
    root_dir = 'outputs_gibran'
    file_generation = '{}/generation_number.npy'.format(root_dir)
    generation = np.load(file_generation)
    fitness_score_list = []
    active_nodes_list = []
    for gen in range(0, generation+1):
        file_pop = '{}/gen{}_pop.npy'.format(root_dir, gen)
        population = np.load(file_pop)
        scores = []
        for individual in population:
            scores.append(individual.fitness.values[0])
        sample_best = population[np.random.choice(a=np.where(np.min(scores)==scores)[0], size=1)[0]]
        # print('Generation: {}'.format(gen))
        # display_genome(sample_best)
        active_nodes = sample_best.skeleton[1]["block_object"].active_nodes
        fitness_score_list.append(1 - sample_best.fitness.values[0])
        active_nodes_list.append(len(active_nodes))
    plt.subplot(2, 1, 1)
    plt.plot(range(0, generation + 1), fitness_score_list, linestyle='--', marker='o', color = 'black')
    plt.legend(['accuracy_score'])
    plt.subplot(2, 1, 2)
    plt.plot(range(0, generation + 1), active_nodes_list, linestyle='--', marker='o', color = 'r')
    plt.legend(['active_nodes length'])
    plt.tight_layout()
    plt.show()

def draw_analysis2():
    reference_point = (1, 1)
    hv = HyperVolume(reference_point)
    root_dir = 'outputs_normed_cifar_more_generations'
    file_generation = '{}/generation_number.npy'.format(root_dir)
    generation = np.load(file_generation)
    accuracy_score_list = []
    f1_score_list = []
    active_nodes_list = []
    volumes = []
    populations = []
    for gen in range(0, generation+1):
        gen_fitnesses = []
        file_pop = '{}/gen{}_pop.npy'.format(root_dir, gen)
        population = np.load(file_pop)
        scores = []
        for individual in population:
            scores.append(individual.fitness.values[0])
            gen_fitnesses.append(individual.fitness.values)
        sample_best = population[np.random.choice(a=np.where(np.min(scores)==scores)[0], size=1)[0]]
        # print('Generation: {}'.format(gen))
        # display_genome(sample_best)
        active_nodes = sample_best.skeleton[1]["block_object"].active_nodes
        accuracy_score_list.append(1 - sample_best.fitness.values[0])
        f1_score_list.append(1 - sample_best.fitness.values[1])
        active_nodes_list.append(len(active_nodes))
        volumes.append(1 - hv.compute(gen_fitnesses))
        populations += list(population)
    plt.subplot(221)
    plt.plot(range(0, generation + 1), accuracy_score_list, linestyle='--', marker='o', color = 'black')
    # plt.legend(['accuracy_score'])
    plt.title('Accuracy over generations')
    plt.ylabel('Accuracy')
    plt.xlabel('Generations')
    plt.subplot(222)
    plt.plot(range(0, generation + 1), active_nodes_list, linestyle='--', marker='o', color = 'r')
    # plt.legend(['active_nodes length'])
    plt.tight_layout()
    plt.title('Active nodes over generations')
    plt.ylabel('Number of active nodes')
    plt.xlabel('Generations')
    plt.subplot(223)
    plt.plot(range(0, generation + 1), volumes, linestyle='--', marker='o', color = 'black')
    # plt.legend(['hyper volume over generations'])
    plt.title('HyperVolume over generations')
    plt.ylabel('HyperVolume')
    plt.xlabel('Generations')
    plt.subplot(224)
    plt.plot(range(0, generation + 1), f1_score_list, linestyle='--', marker='o', color = 'r')
    # plt.legend(['active_nodes length'])
    plt.tight_layout()
    plt.title('F1-score over generations')
    plt.ylabel('F1-score')
    plt.xlabel('Generations')
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # plt.savefig('{}_saved'.format(root_dir))
    plt.show()

    # populations = populations
    all_scores = np.array([indiv.fitness.values for indiv in populations])
    fit_acc = all_scores[:, 0]
    fit_f1 = all_scores[:, 1]
    best_ind = populations[fit_acc.argmin()]
    print('Best individual had fitness: {}'.format(best_ind.fitness.values))
    display_genome(best_ind)

def display_genome(individual):
    print('The genome is: ')
    for i in range(1,individual.num_blocks+1):
        curr_block = individual.skeleton[i]["block_object"]
        print('curr_block isDead = ', curr_block.dead)
        print(curr_block.active_nodes)
        for active_node in curr_block.active_nodes:
            fn = curr_block[active_node]
            # print(fn)
            print('function at: {} is: {}'.format(active_node, fn))

if __name__ == '__main__':
    draw_analysis2()
