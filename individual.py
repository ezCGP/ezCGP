### individual.py

# external packages

# my scripts
from blocks import Block

class Individual(Block):
	"""
	Individual genome composed of blocks of smaller genome.
	Here we define those blocks into the full genome
	"""
	def __init__(self):
		# TODO eventually make it so we can read in an xml/json/yaml file
		# the user will be able to outline the individual.genome there
		# instead of having to edit this file every time
		self.genome = [] #maybe a dict instead?
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
