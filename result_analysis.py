import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def draw_analysis():
    file_generation = 'outputs_mnist/generation_number.npy'
    generation = np.load(file_generation)
    fitness_score_list = []
    active_nodes_list = []
    for gen in range(0, generation+1):
        file_pop = 'outputs_mnist/gen%i_pop.npy' % (gen)
        population = np.load(file_pop)
        scores = []
        for individual in population:
            scores.append(individual.fitness.values[0])
        sample_best = population[np.random.choice(a=np.where(np.min(scores)==scores)[0], size=1)[0]]
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

if __name__ == '__main__':
    draw_analysis()
