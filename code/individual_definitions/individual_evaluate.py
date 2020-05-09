'''
root/individual_definitions/individual_evaluate.py

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
from code.genetic_material import IndividualMaterial
from code.individual_definitions.individual_definition import IndividualDefinition



class IndividualEvaluate_Abstract(ABC):
    '''
    REQUIREMENTS/EXPECTATIONS

    Individual Evaluate class:
     * inputs: instance of IndividualDefinition, an instance of IndividualMaterial, and the training+validation data
     * returns: the direct output of the last block
    '''
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, indiv_def, indiv_material, training_datapair, validation_datapair=None):
        pass



class IndividualEvaluate_Standard(IndividualEvaluate_Abstract):
    def __init__(self):
        pass

    def evaluate(self, indiv_def, indiv, training_datapair, validation_datapair=None):
        for block_index, block in enumerate(indiv_material.blocks):
            block_def = indiv_def[block_index]
            if block.need_evaluate:
                training_datapair = block_def.evaluate(block, training_datapair, validation_datapair)

        indiv_material.output = training_datapair #TODO figure this out