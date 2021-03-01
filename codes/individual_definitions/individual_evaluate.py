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
                 training_datalist: ezData,
                 validating_datalist: ezData=None,
                 supplements=None):
        pass


    def _evaluate(self,
                  indiv_id,
                  block_index,
                  block_def,
                  block_material,
                  training_datalist,
                  validating_datalist=None,
                  supplements=None,
                  apply_deepcopy=True):
        '''
        We've noted that many blocks can have slight variations for what they send to evaluate and how it is received back
        BUT there are still a lot of the same code used in each. So we made this method that should be universal to all
        blocks, and then each class can have their own custom evaluate() method where they use this universal _evaluate()

        Also always true:
            training_datalist, validating_datalist, supplements = block_material.output
        '''
        if apply_deepcopy:
            input_args = [deepcopy(training_datalist), deepcopy(validating_datalist), supplements]
        else:
            input_args = [training_datalist, validating_datalist, supplements]

        if block_material.need_evaluate:
            ezLogging.info("%s - Sending to %ith BlockDefinition %s to Evaluate" % (indiv_id, block_index, block_def.nickname))
            block_def.evaluate(block_material, *input_args)
            if block_material.dead:
                indiv_material.dead = True
            else:
                pass
        else:
            ezLogging.info("%s - Didn't need to evaluate %ith BlockDefinition %s" % (indiv_id, block_index, block_def.nickname))



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
                 training_datalist: ezData,
                 validating_datalist: ezData=None,
                 supplements=None):
        for block_index, (block_material, block_def) in enumerate(zip(indiv_material.blocks, indiv_def.block_defs)):
            self._evaluate(indiv_id,
                           block_index,
                           block_def,
                           block_material,
                           training_datalist,
                           validating_datalist)
            training_datalist, validating_datalist, supplements = block_material.output

        indiv_material.output = block_material.output



class IndividualEvaluate_wAugmentorPipeline_wTensorFlow(IndividualEvaluate_Abstract):
    '''
    Here we are assuming that our datalist will have at least these 2 ezData instances:
        * ezData_Images
        * ezData_Augmentor

    It is also assumed that ezData_Images is HUGE, so we do not want to pass this huge thing
    into every block for evaluation, and so down the road it won't get saved to the individual
    in it's block_material.output
    Instead we only want to pass the ezData_Augmentor to blocks that handle 'preprocessing' or
    'data augmentation'.
    '''
    def __init__(self):
        pass


    def evaluate(self,
                 indiv_material: IndividualMaterial,
                 indiv_def, #IndividualDefinition,
                 training_datalist: ezData,
                 validating_datalist: ezData,
                 supplements=None):
        '''
        We only want to pass in the 'pipeline' of the data if the block does 'data augmentation' or 'data preprocessing'.

        First find the index in datalist for our ezData_Augmentor. Assume the indices are the same for training +
        validating datalists
        '''
        from data.data_tools.ezData import ezData_Augmentor

        augmentor_instance_index = None
        for i, data_instance in training_datalist:
            if isinstance(data_instance, ezData_Augmentor):
                augmentor_instance_index = i
                break
        if augmentor_instance_index is None:
            ezLogging.error("No ezData_Augmentor instance found in training_datalist")
            exit()

        for block_index, (block_material, block_def) in enumerate(zip(indiv_material.blocks, indiv_def.block_defs)):
            if ('augment' in block_def.nickname.lower()) or ('preprocess' in block_def.nickname.lower()):
                temp_training_datalist = [training_datalist[augmentor_instance_index]]
                temp_validating_datalist = [validating_datalist[augmentor_instance_index]]
                self._evaluate(indiv_id,
                               block_index,
                               block_def,
                               block_material,
                               temp_training_datalist,
                               temp_validating_datalist)
                training_datalist[augmentor_instance_index], validating_datalist[augmentor_instance_index], _ = block_material.output

            elif ('tensorflow' in block_def.nickname.lower()) or ('tfkeras' in block_def.nickname.lower()):
                self._evaluate(indiv_id,
                               block_index,
                               block_def,
                               block_material,
                               temp_training_datalist,
                               temp_validating_datalist)
                _, _, indiv_material.output = block_material.output

            else:
                self._evaluate(indiv_id,
                               block_index,
                               block_def,
                               block_material,
                               temp_training_datalist,
                               temp_validating_datalist)
                training_datapair, validation_datapair, _ = block_material.output



class IndividualEvaluate_wAugmentorPipeline_wTransferLearning_wTensorFlow(IndividualEvaluate_Abstract):
    '''
    Similar to IndividualEvaluate_wAugmentorPipeline_wTensorFlow but we are going to pass supplemental info
    between TransferLearning Block and TensorFlow Block ie NN graph, first and last layer of downloaded model, etc
    '''
    def __init__(self):
        pass


    def evaluate(self,
                 indiv_material: IndividualMaterial,
                 indiv_def, #IndividualDefinition,
                 training_datalist: ezData,
                 validating_datalist: ezData,
                 supplements=None):
        '''
        placeholding
        '''
        cannot_pickle_tfkeras = False
        from data.data_tools.ezData import ezData_Augmentor

        augmentor_instance_index = None
        for i, data_instance in training_datalist:
            if isinstance(data_instance, ezData_Augmentor):
                augmentor_instance_index = i
                break
        if augmentor_instance_index is None:
            ezLogging.error("No ezData_Augmentor instance found in training_datalist")
            exit()

        for block_index, (block_material, block_def) in enumerate(zip(indiv_material.blocks, indiv_def.block_defs)):
            if ('augment' in block_def.nickname.lower()) or ('preprocess' in block_def.nickname.lower()):
                temp_training_datalist = [training_datalist[augmentor_instance_index]]
                temp_validating_datalist = [validating_datalist[augmentor_instance_index]]
                self._evaluate(indiv_id,
                               block_index,
                               block_def,
                               block_material,
                               temp_training_datalist,
                               temp_validating_datalist)
                training_datalist[augmentor_instance_index], validating_datalist[augmentor_instance_index], _ = block_material.output

            elif ('transferlearning' in block_def.nickname.lower()) or ('transfer_learning' in block_def.nickname.lower()):
                if (cannot_pickle_tfkeras) & (indiv_material[block_index+1].need_evaluate):
                    # then we had to delete the tf.keras.model in supplements index of block_material.output, so we have to re-eval
                    block_material.need_evaluate = True
                temp_training_datalist = [training_datalist[augmentor_instance_index]]
                temp_validating_datalist = [validating_datalist[augmentor_instance_index]]
                self._evaluate(indiv_id,
                               block_index,
                               block_def,
                               block_material,
                               temp_training_datalist,
                               temp_validating_datalist)
                # place existing tf.keras.model from transfer learning step into supplements
                training_datalist[augmentor_instance_index], validating_datalist[augmentor_instance_index], supplements = block_material.output
                if cannot_pickle_tfkeras:
                    block_material.output[-1] = None

            elif ('tensorflow' in block_def.nickname.lower()) or ('tfkeras' in block_def.nickname.lower()):
                self._evaluate(indiv_id,
                               block_index,
                               block_def,
                               block_material,
                               temp_training_datalist,
                               temp_validating_datalist,
                               supplements,
                               apply_deepcopy=False)
                _, _, indiv_material.output = block_material.output

            else:
                self._evaluate(indiv_id,
                               block_index,
                               block_def,
                               block_material,
                               temp_training_datalist,
                               temp_validating_datalist)
                training_datapair, validation_datapair, _ = block_material.output
