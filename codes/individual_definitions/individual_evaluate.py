'''
root/individual_definitions/individual_evaluate.py

Overview:
Super basic. All we do is define a class that has a single method: evaluate().
The method should take in IndividualMaterial (the thing it needs to evaluate), IndividualDefinition (guide for how to evaluate), and the data.
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
#from codes.individual_definitions.individual_definition import IndividualDefinition #circular dependecy
from codes.utilities.custom_logging import ezLogging



class IndividualEvaluate_Abstract(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self,
                 indiv_material: IndividualMaterial,
                 indiv_def, #: IndividualDefinition,
                 training_datapair: ezDataSet,
                 validation_datapair: ezDataSet=None):
        pass



class IndividualEvaluate_Standard(IndividualEvaluate_Abstract):
    '''
    for loop over each block; evaluate, take the output, and pass that in as the input to the next block
    check for dead blocks (errored during evaluation) and then just stop evaluating. Note, the remaining blocks
    should continue to have the need_evaluate flag as True.
    '''
    def __init__(self):
        pass

    def evaluate(self,
                 indiv_material: IndividualMaterial,
                 indiv_def, #: IndividualDefinition,
                 training_datapair: ezDataSet,
                 validation_datapair: ezDataSet=None):
        for block_index, (block_material, block_def) in enumerate(zip(indiv_material.blocks, indiv_def.block_defs)):
            if block_material.need_evaluate:
                ezLogging.info("%s - Sending to %ith BlockDefinition %s to Evaluate" % (indiv_material.id, block_index, block_def.nickname))
                training_datapair = block_def.evaluate(block_material, training_datapair, validation_datapair)
                if block_material.dead:
                    indiv_material.dead = True
                    break
                else:
                    pass
            else:
                ezLogging.info("%s - Didn't need to evaluate %ith BlockDefinition %s" % (indiv_material.id, block_index, block_def.nickname))
                training_datapair = deepcopy(block_material.output)

        indiv_material.output = training_datapair