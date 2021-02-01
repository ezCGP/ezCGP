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
from data.data_tools.ezData import ezData
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
                 training_datapair: ezData,
                 validation_datapair: ezData=None):
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
                 training_datapair: ezData,
                 validation_datapair: ezData=None):
        for block_index, (block_material, block_def) in enumerate(zip(indiv_material.blocks, indiv_def.block_defs)):
            if block_material.need_evaluate:
                ezLogging.info("%s - Sending to %ith BlockDefinition %s to Evaluate" % (indiv_material.id, block_index, block_def.nickname))
                block_def.evaluate(block_material, deepcopy(training_datapair), deepcopy(validation_datapair))
                if block_material.dead:
                    indiv_material.dead = True
                    break
                else:
                    pass
            else:
                ezLogging.info("%s - Didn't need to evaluate %ith BlockDefinition %s" % (indiv_material.id, block_index, block_def.nickname))
            training_datapair = block_material.output
        
        indiv_material.output = block_material.output



class IndividualEvaluate_withValidation(IndividualEvaluate_Abstract):
    '''
    In IndividualEvaluate_Standard() it is assumed we don't have validation data, so each block
    only outputs the training_datapair for the next block.
    With validation data, we want to return and pass the training and validation data between blocks.
    ...otherwise the process flow is the same
    '''
    def __init__(self):
        pass
    
    
    def evaluate(self,
                 indiv_material: IndividualMaterial,
                 indiv_def, #IndividualDefinition,
                 training_datapair: ezData,
                 validation_datapair: ezData):
        '''
        we want to deepcopy the data before evaluate so that in future if need_evaluate is False, we can grab the
        block_material.output and it will be unique to that block not shared with whole individual.
        '''
        for block_index, (block_material, block_def) in enumerate(zip(indiv_material.blocks, indiv_def.block_defs)):
            if block_material.need_evaluate:
                ezLogging.info("%s - Sending to %ith BlockDefinition %s to Evaluate" % (indiv_material.id, block_index, block_def.nickname))
                block_def.evaluate(block_material, deepcopy(training_datapair), deepcopy(validation_datapair))
                if block_material.dead:
                    indiv_material.dead = True
                    break
                else:
                    pass
            else:
                ezLogging.info("%s - Didn't need to evaluate %ith BlockDefinition %s" % (indiv_material.id, block_index, block_def.nickname))
            training_datapair, validation_datapair = block_material.output

        indiv_material.output = training_datapair, validation_datapair



class IndividualEvaluate_withValidation_andTransferLearning(IndividualEvaluate_Abstract):
    '''
    In IndividualEvaluate_Standard() it is assumed we don't have validation data, so each block
    only outputs the training_datapair for the next block.
    With validation data, we want to return and pass the training and validation data between blocks.
    ...otherwise the process flow is the same

    see note in evaluate() for specifics of Transfer Learning
    '''
    def __init__(self):
        pass
    
    
    def evaluate(self,
                 indiv_material: IndividualMaterial,
                 indiv_def, #IndividualDefinition,
                 training_datapair: ezData,
                 validation_datapair: ezData):
        '''
        we want to deepcopy the data before evaluate so that in future if need_evaluate is False, we can grab the
        block_material.output and it will be unique to that block not shared with whole individual.

        the output of tansfer learning with tfkeras are the first and last layers of the model. those are added to
        attributes of training datapair but that makes the datapair un-deepcopy-able. so we don't deepcopy it,
        but that means that we can't save the state of datapair after transfer learning but before tensorflow block
        so even if need_evaluate is False, we also re-evaluate since the saved output is really the output of tensorflow
        block not after transfer learning.
        '''
        for block_index, (block_material, block_def) in enumerate(zip(indiv_material.blocks, indiv_def.block_defs)):
            if (block_def.nickname == 'transferlearning_block') and (indiv_material[block_index+1].need_evaluate):
                # if the tensorflow_block need_evaluate, then we also need to evaluate the transferlearning_block
                # otherwise we assume that all blocks of individual are need_evaluate False so it'll just grab indiv_material.output
                block_material.need_evaluate = True

            if block_material.need_evaluate:
                ezLogging.info("%s - Sending to %ith BlockDefinition %s to Evaluate" % (indiv_material.id, block_index, block_def.nickname))
                if block_def.nickname == 'tensorflow_block':
                    block_def.evaluate(block_material, training_datapair, validation_datapair)
                    # delete anything we don't need anymore that will break a 'deepcopy' call
                    del training_datapair.graph_input_layer
                    del training_datapair.final_pretrained_layer
                else:
                    block_def.evaluate(block_material, deepcopy(training_datapair), deepcopy(validation_datapair))
                if block_material.dead:
                    indiv_material.dead = True
                    break
                else:
                    pass
            else:
                ezLogging.info("%s - Didn't need to evaluate %ith BlockDefinition %s" % (indiv_material.id, block_index, block_def.nickname))
            training_datapair, validation_datapair = block_material.output

        # training_datapair will be None, and validation_datapair will be the final fitness scores
        indiv_material.output = validation_datapair
