### individual.py

# external packages

# my scripts
from blocks import Block

class Individual(): # Block not inherited in...rather just instanciate to an attribute or something
	"""
	Individual genome composed of blocks of smaller genome.
	Here we define those blocks into the full genome
	"""
	def __init__(self):
		# TODO eventually make it so we can read in an xml/json/yaml file
		# the user will be able to outline the individual.genome there
		# instead of having to edit this file every time
		
		#self.genome = [] #maybe a dict instead?
		"""
		[
		block1 = {"inputs"=[inputs0,inputs1],"args","outputs"},
		block2 = {"inputs"=block1.outputs,"args","outputs"},
		output0 = block2.output[0],
		output1 = block2.output[1],
		inputs1,
		inputs0
		]
		"""
		self.preprocessing = Block(
								nickname="preprocessing",
								ftn_dict={"ftnName": {"prob"1:}},
								arg_dict={"argumentDataType": {"prob":1}},
								mate_dict={"mateMethod": {"prob":1, "args":[]}},
								mut_dict={"mutateMethod": {"prob":1, "args":[]}},
								gene_dict={ftnName: {"inputs":[], "args":[], "output":_}},
								block_inputs=[inputDataType, inputDataType],
								block_outputs=[outputDataType, outputDataType],
								block_main_count=num_main,
								block_arg_count=num_args)

		self.tensorblock_classify = Block(...)


	def evaluate(self, genome_inputs):
		self.preprocessing.evaluate(
							block_inputs=genome_inputs,
							learning_required=False)
		self.tensorblock_classify.evaluate(
							block_inputs=self.preprocessing.block_outputs_values,
							learning_required=False)
		self.genome_outputs = self.tensorblock_classify.block_outputs_values
		self.score_fitness()
		#return genome_outputs

	def score_fitness(self, labels):
		#self.fitness = 
		pass


	def mutate(self, block=None):
		if block is None:
			self.preprocessing.mutate()
			self.tensorblock_classify.mutate()
		elif block is self.tensorblock_classify.name:
			# TODO we need a better way to be able to come up with these blocks in a list or something
			self.tensorblock_classify.mutate()
		elif block is self.preprocessing.name:
			self.preprocessing.mutate()
		elif block is "random selection":
			roll = np.random.random()
			if roll < 0.5:
				#mutate one...
		else:
			print("UserError: unrecognized block name: %s" % block)
			pass


	def mate(self, block=None):
		# mate both or only one?
		pass
		# return offspring_list