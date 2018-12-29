### universe.py
# a single instnace of a universe...unique 'state of the world'

# external packages
import numpy as np

# my scripts
from individual import Individual

def run_universe(seed, population_size):
	np.random.seed(seed)

	# initialize the population
	population = []
	for i in range(population_size):
		individual = Individual()