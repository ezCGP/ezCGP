'''
noticed that Augmentor.Pipeline() instance was maintaining the operators from the previous individual
who was evaluated, even after deepcopying the data before passing it to block_def.evaluate()...why?

this is an investigation into that
'''
### packages
import os
import numpy as np
from copy import deepcopy
import pdb
import Augmentor

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from data.data_tools import ezData
from codes.block_definitions.utilities import operators_Augmentor_augmentation as aug_ops

### create fake data
class ezData_Fake(ezData.ezData):
	'''
	fake class to look like the data we would use.
	throw some fake attributes in here along with the pipeline and 
	see what happens when we deepcopy
	'''
	def __init__(self):
		self.x = np.random.random((100,3))
		self.y = np.arange(100)
		self.pipeline = Augmentor.Pipeline()



class ezData_FakeWrapper(ezData.ezData):
	'''
	make it have wrapper attributes like with cifar10 for images + pipeline
	'''
	def __init__(self):
		self.pipeline_wrapper = ezData_Fake()


### mimic evaluation calls
def block_evaluate(ezdata):
	ezdata.pipeline = aug_ops.rotate(ezdata.pipeline)
	return ezdata


def individual_evaluate(ezdata):
	for ith_block in range(3):
		ezdata = block_evaluate(deepcopy(ezdata))
	return ezData


def individual_evaluate_wWrapper(ezdata):
	for ith_block in range(3):
		ezdata.pipeline_wrapper = block_evaluate(deepcopy(ezdata.pipeline_wrapper))
	return ezdata



### set up experiments
def test_1(myinput):
	'''
	test only block_evaluate with deepcopy
	'''
	print("Test 1 - Block Evaluate")
	myoutput = block_evaluate(deepcopy(myinput))
	print("input operators: %i" % (len(myinput.pipeline.operations)))


def test_2(myinput):
	'''
	mimic full evaluate starting with individaul_evaluate
	'''
	print("Test 2 - Individual Evaluate")
	myoutput = individual_evaluate(myinput)
	print("input operators: %i" % (len(myinput.pipeline.operations)))


def test_3(myinput):
	'''
	mimic full evaluate starting with individaul_evaluate
	'''
	print("Test 3 - Individual Evaluate w/ Wrapper")
	myoutput = individual_evaluate_wWrapper(myinput)
	print("input operators: %i" % (len(myinput.pipeline_wrapper.pipeline.operations)))



original_1 = ezData_Fake()
test_1(original_1)

original_2 = ezData_Fake()
test_2(original_2)

original_3 = ezData_FakeWrapper()
test_3(original_3) ### ef my anus