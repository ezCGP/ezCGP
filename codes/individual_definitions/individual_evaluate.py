'''
root/individual_definitions/individual_evaluate.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code


Rules:
always have a check for dead blocks
'''

### packages
from copy import deepcopy
from abc import ABC, abstractmethod

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from data.data_tools.data_types import ezDataSet
from codes.genetic_material import IndividualMaterial
from codes.individual_definitions.individual_definition import IndividualDefinition



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
    def evaluate(self,
                 indiv_material: IndividualMaterial,
                 indiv_def: IndividualDefinition,
                 training_datapair: ezDataSet,
                 validation_datapair: ezDataSet=None):
        pass



class IndividualEvaluate_Standard(IndividualEvaluate_Abstract):
    def __init__(self):
        pass

    def evaluate(self,
                 indiv_material: IndividualMaterial,
                 indiv_def: IndividualDefinition,
                 training_datapair: ezDataSet,
                 validation_datapair: ezDataSet=None):
        for block_index, block_material in enumerate(indiv_material.blocks):
            block_def = indiv_def[block_index]
            if block_material.need_evaluate:
                training_datapair = block_def.evaluate(block_material, training_datapair, validation_datapair)
                if block_material.dead:
                    indiv_material.dead = True
                    break
                else:
                    pass
            else:
                training_datapair = deepcopy(block_material.output)

        indiv_material.output = training_datapair #TODO figure this out