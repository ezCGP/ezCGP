'''
mimic what run_ea would do
'''

### packages
import os
import numpy as np

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
import evolve_only_module as ezcgp_module


def fake_selection(population):
	'''
	excpet population to be a list of 2-element tuples but we only care about first element
	...well actually we don't care at all since this is all fake.
	'''
	next_pop_size = int(len(population)*0.7)
	next_pop_size -= next_pop_size%4 # now make it divisible by for so we can do parent_selection
	next_pop_index = np.random.choice(np.arange(len(population)), next_pop_size, replace=False)
	next_pop = []
	for i in next_pop_index:
		next_pop.append(population[i])

	return next_pop


def main(config, problem, seed=9):
	ezcgp = ezcgp_module.ezCGP_Module(config, problem, seed)
	population = None
	generation_limit = 3
	for i in range(generation_limit):
		print("\nGENERATION %02i" % i)
		population = ezcgp.run(population);# import pdb; pdb.set_trace()

		# check validity of each
		sys.path.append(ezcgp.universe_output_directory)
		for mdf_file, indiv_id in population:
			mdf_file = os.path.basename(mdf_file)
			mdf_module = __import__(mdf_file[:-3])

			mdf = mdf_module.My_MDF()
			output = mdf.temp_fake_function(3, 2)
			print(mdf_file, indiv_id, output)

		population = fake_selection(population)

	ezcgp.close()


if __name__ == "__main__":
	ezcgp_root_dir = "/home/roddtalebi/Documents/ezCGP/" # unique for each user
	config_filepath = os.path.join(ezcgp_root_dir, "misc/dynamic_problem_supplements/example_mdf_config.json")
	problem_filepath = os.path.join(ezcgp_root_dir, "problems/problem_MDF.py")

	main(config_filepath, problem_filepath)
