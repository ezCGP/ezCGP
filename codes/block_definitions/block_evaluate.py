'''
root/block_definitions/block_evaluate.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
from abc import ABC, abstractmethod

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from data.data_tools.data_types import ezDataSet
from codes.genetic_material import BlockMaterial
from codes.block_definitions.block_definition import BlockDefinition



class BlockEvaluate_Abstract(ABC):
    '''
    REQUIREMENTS/EXPECTATIONS

    Block Evaluate class:
     * should start with a reset_evaluation() method
     * inputs: an instance of BlockMaterial, and the training+validation data
     * returns: a list of the output to the next block or as output of the individual
    '''
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, block_material, training_datapair, validation_datapair=None):
        pass

    @abstractmethod
    def reset_evaluation(self, block_material):
        pass



class BlockEvaluate_GraphAbstract(BlockEvaluate_Abstract):
    '''
    attempt at abstracting what an EvaluateDefinition will look like for a 
    computational graph block like tensorflow, pytorch, or keras

    these are just ideas
    
    Edit notes (Sam): TF 2.0 has a tf.function class that builds computational graphs automatically (is recommended), see operators.py
    '''
    @abstractmethod
    def build_graph(self):
        pass

    @abstractmethod
    def reset_graph(self):
        pass

    @abstractmethod
    def train_graph(self):
        pass

    @abstractmethod
    def run_graph(self):
        pass



class BlockEvaluate_Standard(BlockEvaluate_Abstract):
    '''
    TODO
    '''
    def evaluate(self, block_def, block_material, training_datapair, validation_datapair=None):
        self.reset_evaluation(block_material)

        # add input data
        for i, data_input in enumerate(training_datapair):
            block_material.evaluated[-1*(i+1)] = data_input

        # go solve
        for node_index in block_material.active_nodes:
            if node_index < 0:
                # do nothing. at input node
                continue
            elif node_index >= block_material.main_count:
                # do nothing NOW. at output node. we'll come back to grab output after this loop
                continue
            else:
                # main node. this is where we evaluate
                function = block_material[node_index]["ftn"]
                
                inputs = []
                node_input_indices = block_material[node_index]["inputs"]
                for node_input_index in node_input_indices:
                    inputs.append(block_material.evaluated[node_input_index])

                args = []
                node_arg_indices = block_material[node_index]["args"]
                for node_arg_index in node_arg_indices:
                    args.append(block_material.args[node_arg_index].value)

                #print(function, inputs, args)
                block_material.evaluated[node_index] = function(*inputs, *args)
                '''try:
                    self.evaluated[node_index] = function(*inputs, *args)
                except Exception as e:
                    print(e)
                    self.dead = True
                    break'''

        output = []
        for output_index in range(block_def.main_count, block_def.main_count+block_def.output_count):
            output.append(block_material.evaluated[block_material.genome[output_index]])

        return output


    def reset_evaluation(self, block_material):
        block_material.evaluated = [None] * len(block_material.genome)
        block_material.output = None
        block_material.dead = False