import numpy as np
import os
import matplotlib.pyplot as plt
from hypervolume import HyperVolume

import sys
from matplotlib.animation import FuncAnimation

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
        active_nodes = sample_best.skeleton[sample_best.num_blocks]["block_object"].active_nodes
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
    root_dir = 'outputs_cifar_augment'
    file_generation = '{}/generation_number.npy'.format(root_dir)
    generation = np.load(file_generation)
    accuracy_score_list = []
    f1_score_list = []
    active_nodes_list = []
    volumes = []
    populations = []

    paretoInds = []  # list to keep track of current pareto dominant inds
    paretoScoreList = []  # keeps track of pareto scores through time
    for gen in range(0, generation+1):
        gen_fitnesses = []
        file_pop = '{}/gen{}_pop.npy'.format(root_dir, gen)
        population = np.load(file_pop)
        scores = []
        for individual in population:
            scores.append(individual.fitness.values[0])
            gen_fitnesses.append(individual.fitness.values)
        
        megaPopulation = population + paretoInds  # combine population and pareto inds temporarily
        paretoInds = []
        for i in range(len(megaPopulation)):
            candidate_acc = 1 - megaPopulation[i].fitness.values[0]
            candidate_f1 = 1- megaPopulation[i].fitness.values[1]
            dominated = False
            for i_ in range(len(megaPopulation)):
                if i != i_:
                    comp_acc = 1 - megaPopulation[i_].fitness.values[0]
                    comp_f1 = 1 - megaPopulation[i_].fitness.values[1]
                    if comp_acc > candidate_acc and comp_f1 > candidate_f1:
                        dominated = True
            if not dominated:
                paretoInds.append(megaPopulation[i])

        paretoScoreList.append([ind.fitness.values for ind in paretoInds])



        sample_best = population[np.random.choice(a=np.where(np.min(scores)==scores)[0], size=1)[0]]
        # print('Generation: {}'.format(gen))
        # display_genome(sample_best)
        active_nodes = sample_best.skeleton[sample_best.num_blocks]["block_object"].active_nodes
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



        # pareto front graphs
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    # Query the figure's on-screen size and DPI. Note that when saving the figure to
    # a file, we need to provide a DPI for that separately.
    print('fig size: {0} DPI, size in inches {1}'.format(
        fig.get_dpi(), fig.get_size_inches()))


    def update(paretoScores):
        acc_score = [i[0] for i in paretoScores]
        f1_score = [i[1] for i in paretoScores]
        ax.plot(acc_score, f1_score)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        line.set_ydata(x - 5 + i)
        #ax.set_xlabel(label)
        return line, ax

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=paretoScoreList, interval=200)
    anim.save('line.gif', dpi=80, writer='imagemagick')
    plt.show()

def display_genome(individual):
    print('The genome is: ')
    for i in range(1,individual.num_blocks+1):
        curr_block = individual.skeleton[i]["block_object"]
        print('curr_block isDead = ', curr_block.dead)
        print(curr_block.active_nodes)
        arg_values = np.array(curr_block.args)
        for active_node in curr_block.active_nodes:
            fn = curr_block[active_node]
            if active_node < 0:
                # nothing to evaluate at input nodes
                print('function at: {} is: {}'\
                    .format(active_node, fn))
                continue
            elif active_node >= curr_block.genome_main_count:
                # nothing to evaluate at output nodes
                print('function at: {} is: {} -> likely an output node'\
                    .format(active_node, fn))
                continue
            print('function at: {} is: {} and has arguments: {}'\
                    .format(active_node, fn, arg_values[fn['args']]))

if __name__ == '__main__':
    draw_analysis2()
