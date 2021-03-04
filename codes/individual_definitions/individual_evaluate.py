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
from data.data_tools.simganData import SimGANDataset
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



class IndividualEvaluate_withValidation_andTransferLearning_DEPRECIATED(IndividualEvaluate_Abstract):
    '''
    Pass both training + validation data through the blocks.

    With transfer learning, we are passing layers of the TFKeras graph between blocks which
    makes them unable to be deepcopied; so for those blocks we don't deepcopy the data, so 
    we can't save the state of the output of that block so it always has to have need_evaluate
    to True.

    We also are assuming that our data is really a wrapper around 2 other data objects: one
    for the pipline and one for the actual images and only part of that get's passed through
    the blocks.
    '''
    def __init__(self):
        pass
    
    
    def evaluate(self,
                 indiv_material: IndividualMaterial,
                 indiv_def, #IndividualDefinition,
                 training_datapair: ezData,
                 validation_datapair: ezData):

        for block_index, (block_material, block_def) in enumerate(zip(indiv_material.blocks, indiv_def.block_defs)):
            if (block_def.nickname == 'transferlearning_block') and (indiv_material[block_index+1].need_evaluate):
                # if the tensorflow_block need_evaluate, then we also need to evaluate the transferlearning_block
                # otherwise we assume that all blocks of individual are need_evaluate False so it'll just grab indiv_material.output
                block_material.need_evaluate = True



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


    def evaluate_block(self,
                       indiv_id,
                       block_index,
                       block_def,
                       block_material,
                       training_datapair,
                       validation_datapair):
        '''
        since each block has a slightly different behavior about what exactly get's passed in as data,
        I made a generalized evaluate method here that can get called in several different ways in the
        other evaluate method
        '''
        if block_material.need_evaluate:
            ezLogging.info("%s - Sending to %ith BlockDefinition %s to Evaluate" % (indiv_id, block_index, block_def.nickname))
            if block_def.nickname == 'tensorflow_block':
                # don't deepcopy, just work off the same data-instances from transferlearning block
                block_def.evaluate(block_material, training_datapair, validation_datapair)
                # delete attributes we don't need that can't be "deepcopy"-ed
                del training_datapair.pipeline_wrapper.graph_input_layer
                del training_datapair.pipeline_wrapper.final_pretrained_layer
            else:
                block_def.evaluate(block_material, deepcopy(training_datapair), deepcopy(validation_datapair))
            if block_material.dead:
                indiv_material.dead = True
            else:
                pass
        else:
            ezLogging.info("%s - Didn't need to evaluate %ith BlockDefinition %s" % (indiv_id, block_index, block_def.nickname))
    
    
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
            if ('augment' in block_def.nickname.lower()) or ('preprocess' in block_def.nickname.lower()):
                '''
                then we only want to pass in the 'pipeline' of the data so all the images don't get dragged along
                and also get saved as the output of the block
                '''
                self.evaluate_block(indiv_material.id,
                                    block_index,
                                    block_def,
                                    block_material,
                                    training_datapair.pipeline_wrapper,
                                    validation_datapair.pipeline_wrapper)
                training_datapair.pipeline_wrapper, validation_datapair.pipeline_wrapper = block_material.output

            elif ('transferlearning' in block_def.nickname.lower()) or ('transfer_learning' in block_def.nickname.lower()):
                if (indiv_material[block_index+1].need_evaluate):
                    '''
                    if the tensorflow_block need_evaluate, then we also need to evaluate the transferlearning_block
                    otherwise we assume that all blocks of individual are need_evaluate False so it'll just grab indiv_material.output
                    '''
                    block_material.need_evaluate = True

                self.evaluate_block(indiv_material.id,
                                    block_index,
                                    block_def,
                                    block_material,
                                    training_datapair.pipeline_wrapper,
                                    validation_datapair.pipeline_wrapper)
                training_datapair.pipeline_wrapper, validation_datapair.pipeline_wrapper = block_material.output

            else:
                # must be tensorflow block
                self.evaluate_block(indiv_material.id,
                                    block_index,
                                    block_def,
                                    block_material,
                                    training_datapair,
                                    validation_datapair)

                # training_datapair will be None, and validation_datapair will be the final fitness scores
                _, indiv_material.output = block_material.output


#TODO: customize this
class IndividualEvaluate_SimGAN(IndividualEvaluate_Abstract):
    '''
    for loop over each block; evaluate, take the output, and pass that in as the input to the next block
    check for dead blocks (errored during evaluation) and then just stop evaluating. Note, the remaining blocks
    should continue to have the need_evaluate flag as True.
    '''
    def __init__(self):
        pass

    def evaluate_block(self,
                       indiv_material,
                       block_index,
                       block_def,
                       block_material,
                       train_data,
                       validation_data):
        '''
        Generalized evaluate method since 
        '''
        if block_material.need_evaluate:
            ezLogging.info("%s - Sending to %ith BlockDefinition %s to Evaluate" % (indiv_material.id, block_index, block_def.nickname))
            block_def.evaluate(block_material, train_data, validation_data)
            if block_material.dead:
                indiv_material.dead = True
                return
            else:
                pass
        else:
            ezLogging.info("%s - Didn't need to evaluate %ith BlockDefinition %s" % (indiv_material.id, block_index, block_def.nickname))

    def evaluate(self,
                 indiv_material: IndividualMaterial,
                 indiv_def, #: IndividualDefinition,
                 train_data: SimGANDataset,
                 validation_data: SimGANDataset=None):
        '''
        Because the Refiner and Discriminator are two seperate blocks but require one another for their loss functions, they must be run together
        So we will take the refiner graph and input it into the discriminator block and train/evaluate it there.
        '''

        for block_index, (block_material, block_def) in enumerate(zip(indiv_material.blocks, indiv_def.block_defs)):
            if (block_def.nickname == 'refiner_block') and (indiv_material[block_index+1].need_evaluate):
                # If we are in the refiner block and the discriminator block needs to be reevaluated, then we also need to reevalute the refiner
                block_material.need_evaluate = True
                self.evaluate_block(indiv_material,
                                    block_index,
                                    block_def,
                                    block_material,
                                    train_data,
                                    validation_data)
                train_data, validation_data, refiner = block_material.output
                train_data = (train_data, refiner) # Pack refiner into train data, hacky but allows us to pass in refiner easily to discriminator
            else:
                # must be tensorflow block
                self.evaluate_block(indiv_material,
                                    block_index,
                                    block_def,
                                    block_material,
                                    train_data,
                                    validation_data)
                train_data, validation_data = block_material.output

            last_output = block_material.output
        
        indiv_material.output = last_output
