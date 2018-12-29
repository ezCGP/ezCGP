### main.py

# external packages
from copy import deepcopy

# my scripts
import universe

if __name__ == '__main__':
	# Read in Data

	final_populations = [] # one for each universe created
	for i in range(num_universes):
		converged_solution = universe.run_universe()
		final_populations.append(converged_solution)

		# post processing step for that run