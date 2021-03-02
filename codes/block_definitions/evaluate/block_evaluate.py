'''
root/codes/block_definitions/evaluate/block_evaluate.py

Overview:
Here we define how our block will be 'evaluated'...Of course there is typical concept of evaluating where we just apply methods to data and we're done; then there is 'evaluation' when we are dealing with neural networks where we have to build a graph, train it, and then evaluate it against a different dataset; there have also been cases where we pass through an instantiated class object through the graph and each primitive addes or changes an attribute so evaluation is decorating a class object. This may change in the future, but right now we have generalized the inputs for evaluation to:
* block_material to get the genome and args
* block_def to get metadata about the block
* training and validating data

Here we have 2 methods: evaluate() and reset_evaluation(). We expect the BlockDefinition.evaluate() to run reset_evaluation() and then run evaluate().

CopyPaste from individual_evaluate:
A coding law we use is that blocks will take in and output these 3 things:
    * training_datalist
    * validating_datalist
    * supplements
Sometimes those things can be None, but they should still always be used.
Training + validating datalist are mostly used for when we have multiple blocks and we want to pass
the same data types from one block to the next.
The exception comes at the last block; we mostly aways assume that we no longer car about the datalist,
and only want what is in supplements.
'''

### packages
from abc import ABC, abstractmethod
from copy import deepcopy
import importlib

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from data.data_tools.ezData import ezData
from codes.genetic_material import BlockMaterial
#from codes.block_definitions.block_definition import BlockDefinition #circular dependecy
from codes.utilities.custom_logging import ezLogging



class BlockEvaluate_Abstract(ABC):
    @abstractmethod
    def __init__(self):
        pass


    @abstractmethod
    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def,#: BlockDefinition,
                 training_datalist: ezData,
                 validating_datalist: ezData=None,
                 supplements=None):
        pass


    def standard_evaluate(self,
                          block_material: BlockMaterial,
                          block_def,
                          input_datalist: ezData):
        '''
        After a while of developing, we noticed that ALL our blocks followed the same eval process.
        the main difference was WHAT data was passed in...that's it!
        So this is going to be here (separate from 'evaluate()' as a quick-call method that can
        be used where needed.
        '''
        # verify that the input data matches the expected datatypes
        for input_dtype, input_data in zip(block_def.input_dtypes, input_datalist):
            if input_dtype != type(input_data):
                ezLogging.critical("%s - Input data type (%s) doesn't match expected type (%s)" % (block_material.id, type(input_data), input_dtype))
                return None

        # add input data
        for i, data_input in enumerate(input_datalist):
            block_material.evaluated[-1*(i+1)] = data_input

        # go solve
        for node_index in block_material.active_nodes:
            if node_index < 0:
                # do nothing. at input node
                continue
            elif node_index >= block_def.main_count:
                # do nothing NOW. at output node. we'll come back to grab output after this loop
                continue
            else:
                # main node. this is where we evaluate
                function = block_material[node_index]["ftn"]

                inputs = []
                node_input_indices = block_material[node_index]["inputs"]
                for node_input_index in node_input_indices:
                    inputs.append(block_material.evaluated[node_input_index])
                ezLogging.debug("%s - Eval %i; input index: %s" % (block_material.id, node_index, node_input_indices))

                args = []
                node_arg_indices = block_material[node_index]["args"]
                for node_arg_index in node_arg_indices:
                    args.append(block_material.args[node_arg_index].value)
                ezLogging.debug("%s - Eval %i; arg index: %s, value: %s" % (block_material.id, node_index, node_arg_indices, args))

                ezLogging.debug("%s - Eval %i; Function: %s, Inputs: %s, Args: %s" % (block_material.id, node_index, function, inputs, args))
                try:
                    block_material.evaluated[node_index] = function(*inputs, *args)
                    ezLogging.info("%s - Eval %i; Success" % (block_material.id, node_index))
                except Exception as err:
                    ezLogging.critical("%s - Eval %i; Failed: %s" % (block_material.id, node_index, err))
                    block_material.dead = True
                    import pdb; pdb.set_trace()
                    break

        output_list = []
        if not block_material.dead:
            for output_index in range(block_def.main_count, block_def.main_count+block_def.output_count):
                output_list.append(block_material.evaluated[block_material.genome[output_index]])

        ezLogging.info("%s - Ending standard_evaluate...%i output" % (block_material.id, len(output_list)))
        return output_list


    def preprocess_block_evaluate(self, block_material):
        '''
        should always happen before we evaluate...should be in BlockDefinition.evaluate()

        Note we can always customize this to our block needs which is why we included in BlockEvaluate instead of BlockDefinition
        '''
        ezLogging.debug("%s - Reset for Evaluation" % (block_material.id))
        block_material.output = None
        block_material.evaluated = [None] * len(block_material.genome)
        block_material.dead = False


    def postprocess_block_evaluate(self, block_material):
        '''
        should always happen after we evaluate. important to blow away block_material.evaluated to clear up memory

        can always customize this method which is why we included it in BlockEvaluate and not BlockDefinition
        '''
        ezLogging.debug("%s - Processing after Evaluation" % (block_material.id))
        block_material.evaluated = None
        block_material.need_evaluate = False



class BlockEvaluate_MiddleBlock(BlockEvaluate_Abstract):
    '''
    For a middle-block, we assume that the output of the block is the input of the next block,
    so we replace it with training/validating_datalist instead of passing it through supplements.
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_MiddleBlock Class" % (None, None))


    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def,#: BlockDefinition,
                 training_datalist: ezData,
                 validating_datalist: ezData,
                 supplements=None):
        ezLogging.info("%s - Start evaluating..." % (block_material.id))

        training_datalist = self.standard_evaluate(block_material, block_def, training_datalist)
        if validating_datalist is not None:
            self.preprocess_block_evaluate(block_material) # reset evaluation attributes for validating
            training_datalist = self.standard_evaluate(block_material, block_def, validating_datalist)

        supplements = None
        block_material.output = (training_datalist, validating_datalist, supplements)



class BlockEvaluate_FinalBlock(BlockEvaluate_Abstract):
    '''
    Unlike BlockEvaluate_MiddleBlock, we pass the output of evaluate() to supplements
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_FinalBlock Class" % (None, None))


    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def,#: BlockDefinition,
                 training_datalist: ezData,
                 validating_datalist: ezData,
                 supplements=None):
        ezLogging.info("%s - Start evaluating..." % (block_material.id))

        supplements = []
        supplements.append(self.standard_evaluate(block_material, block_def, training_datalist))
        if validating_datalist is not None:
            supplements.append(self.standard_evaluate(block_material, block_def, validating_datalist))
        else:
            supplements.append(None)

        block_material.output = (None, None, supplements)



class BlockEvaluate_MiddleBlock_SkipValidating(BlockEvaluate_Abstract):
    '''
    Say for data augmentation; this is a step useful in improving your datasete for training,
    so it is useless to use on validating data
    '''
    def __init__(self):
        super().__init__()
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_MiddleBlock_SkipValidating Class" % (None, None))


    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def, #: BlockDefinition,
                 training_datalist: ezData,
                 validating_datalist: ezData,
                 supplements=None):
        ezLogging.info("%s - Start evaluating..." % (block_material.id))

        training_datalist = self.standard_evaluate(block_material, block_def, training_datalist)
        supplements = None
        block_material.output = (training_datalist, validating_datalist, supplements)


'''
class BlockEvaluate_TrainValidate(BlockEvaluate_Standard):
    ''
    In BlockEvaluate_Standard.evaluate() we only evaluate on the training_datalist.
    But here we want to evaluate both training and validating.
    The process flow will be almost identical otherwise.
    ''
    def __init__(self):
        super().__init__()
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_TrainValidate Class" % (None, None))


    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def,#: BlockDefinition,
                 training_datalist: ezData,
                 validating_datalist: ezData,
                 supplements=None):
        ezLogging.info("%s - Start evaluating..." % (block_material.id))

        # going to treat training + validating as separate block_materials!
        output = []
        for datalist in [training_datalist, validating_datalist]:
            single_output_list = self.standard_evaluate(block_material, block_def, [datalist.pipeline])
            datalist.pipeline = single_output_list[0]
            if block_material.dead:
                return []
            else:
                output.append(datalist)
                self.preprocess_block_evaluate(block_material) #prep for next loop through datalist

        block_material.output = output
'''