### universe.py
# a single instnace of a universe...unique 'state of the world'

# external packages
import numpy as np
from copy import deepcopy

# my scripts
from individual import Individual


def run_universe(population, block=None): # be able to select which block we want to evolve or randomly select
	# mate through the population
	mating_list = tournamentSelection(population, k=len(population)) #double check that k works
	for i in range(0,len(mating_list),2):
		parent1 = deepcopy(mating_list[i])
		parent2 = deepcopy(mating_list[i+1])
		offspring_list = parent1.mate(parent2, block)
		for offspring in offspring_list:
			population.append(offspring)

	# mutate through the population
	for individual in population:
		individual.mutate(block)

	# evaluate the population
	for individual in population:
		if individual.need_evaluate:
			individual.evaluate(input_data, block)
			individual.score_fitness(labels)

	return population


def create_universe(input_data, labels, population_size=100, seed=9):
	np.random.seed(seed)

	# initialize the population
	population = []
	for i in range(population_size):
		individual = Individual()
		individual.evaluate(genome_inputs=input_data)
		individual.score_fitness(labels=labels)
		population.append(individual)

	generation = 0
	converged = False
	while (not converged) & (generation<=GENERATION_LIMIT):
		generation += 1
		population = run_universe(population)
		# population multiobjective ranking here or right before it get's returned?

		scores = []
		for individual in population:
			score.append(population.fintess)
		if max(scores) > SCORE_LIMIT:
			converged = True
		else:
			pass
