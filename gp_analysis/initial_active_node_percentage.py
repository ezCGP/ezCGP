'''
root/gp_analysis/*py
initialize genomes of different sizes to get a distribution of active nodes to genome size
'''

### packages
import numpy as np
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import os
# try to fit the data:
from scipy import stats

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root


def init_genome(genome_size, num_inputs=1):
	'''
	assuming all functions only have 1 input
	'''
	# init genome
	genome = [None]*(genome_size+num_inputs+1)
	
	# fill in genome
	genome[-1*num_inputs:] = "inputs"
	for node_index in range(genome_size):
		choices = np.arange(-1*num_inputs, node_index)
		input_node = np.random.choice(choices)
		genome[node_index] = {"inputs": [input_node]}
	output_node = np.random.choice(np.arange(0,genome_size))
	genome[genome_size] = output_node

	# get actives...here we don't count the input + output nodes
	# will remove output node later
	actives = [output_node]
	for node_index in reversed(range(genome_size)):
		if node_index in actives:
			# combine lists
			for input_node in genome[node_index]["inputs"]:
				if input_node >= 0:
					actives.append(input_node)

	# make into set to remove duplicates and then remove output node
	actives = list(set(actives))
	actives.remove(output_node)

	return genome, actives

# save location
root = dirname(dirname(realpath(__file__)))
output_dir = os.path.join(root,
						  "outputs",
						  "initial_actives_node_percentage",
						  time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(output_dir, exist_ok=False)



seed = 9
np.random.seed(seed)

min_size = 5
max_size = 50
for genome_size in range(min_size, max_size+1):
	num_actives = []
	start_time = time.time()
	for i in range(100000):
		_, actives = init_genome(genome_size)
		num_actives.append(len(actives))
	print("genome size %i done after %.2fsec" % (genome_size, time.time()-start_time))

	num_actives = np.array(num_actives)

	'''
	note on plt.hist(density=True) or normed=True
	https://github.com/matplotlib/matplotlib/issues/10398

	does pdf, not just norm so do this instead
	weights=np.ones(len(data))/len(data),density=False
	'''

	fig = plt.figure()
	n, bins, patches = plt.hist(num_actives,
								bins=np.arange(-1,genome_size+1)+0.5,
								weights=np.ones(len(num_actives))/len(num_actives),
								density=False)
	assert(np.abs(1-n.sum()) < 1e-3), "'by count' wasn't normalized right"
	# TODO
	# try fitting to a weibull distribution using scipy
	xs = np.arange(0,10+1,.01) # 0 to max of xlim
	for model in [stats.exponweib,
				  stats.weibull_min,
				  stats.betaprime,
				  #stats.chisquare,
				  stats.f,
				  stats.gamma,
				  stats.rayleigh]:
		plt.plot(xs, model.pdf(xs, *model.fit(num_actives)), label=model.name)
	plt.legend()
	plt.xlim(0,10)
	plt.ylim(0,1)
	plt.title("Histogram by Genome Size\ngenome size %i" % genome_size)
	plt.savefig(os.path.join(output_dir, "bycount_%02d.jpg" % genome_size))
	plt.close()

	fig = plt.figure()
	n, bins, patches = plt.hist(num_actives/genome_size*100,
								bins=np.arange(0,101),
								weights=np.ones(len(num_actives))/len(num_actives),
								density=False)
	assert(np.abs(1-n.sum()) < 1e-3), "'by perc' wasn't normalized right"
	plt.xlim(0,60)
	plt.ylim(0,0.6)
	plt.title("Histogram by Perc of Genome\ngenome size %i" % genome_size)
	plt.savefig(os.path.join(output_dir, "byperc_%02d.jpg" % genome_size))
	plt.close()


