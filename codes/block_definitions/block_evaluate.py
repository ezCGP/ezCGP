'''
root/codes/block_definitions/block_evaluate.py

Overview:
Here we define how our block will be 'evaluated'...Of course there is typical concept of evaluating where we just apply methods to data and we're done; then there is 'evaluation' when we are dealing with neural networks where we have to build a graph, train it, and then evaluate it against a different dataset; there have also been cases where we pass through an instantiated class object through the graph and each primitive addes or changes an attribute so evaluation is decorating a class object. This may change in the future, but right now we have generalized the inputs for evaluation to:
* block_material to get the genome and args
* block_def to get metadata about the block
* training and validation data

Here we have 2 methods: evaluate() and reset_evaluation(). We expect the BlockDefinition.evaluate() to run reset_evaluation() and then run evaluate().
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
                 training_datapair: ezDataSet,
                 validation_datapair: ezDataSet=None):
        pass

    
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
    This could be used for any basic application of methods onto data, like symbolic regression.
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_Standard Class" % (None, None))
        
        
    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def,#: BlockDefinition, 
                 training_datapair: ezDataSet,
                 validation_datapair: ezDataSet=None):
        ezLogging.info("%s - Start evaluating..." % (block_material.id))
        
        # add input data
        for i, data_input in enumerate(training_datapair):
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
                    ezLogging.info("%s - Eval %i; Failed: %s" % (block_material.id, node_index, err))
                    block_material.dead = True
                    import pdb; pdb.set_trace()
                    break

        output = []
        if not block_material.dead:
            for output_index in range(block_def.main_count, block_def.main_count+block_def.output_count):
                output.append(block_material.evaluated[block_material.genome[output_index]])
                
        block_material.output = output
        ezLogging.info("%s - Ending evaluating...%i output" % (block_material.id, len(output)))


    def reset_evaluation(self, block_material: BlockMaterial):
        '''
        we really could just remove this method...I think we're
        keeping it here as a reminder that we can change it
        '''
        super().reset_evaluation(block_material)



class BlockEvaluate_DataAugmentation(BlockEvaluate_Standard):
    '''
    the primitives are unique but the evaluation methods shouldn't be unique, so
    just opting to import BlockEvaluate_Standard
    
    NOTE that we want to add augmentation methods to the 'training data pipeline'
        BUT not to the 'validation/testing data pipeline'
    '''
    def __init__(self):
        super().__init__()
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_DataAugmentation Class" % (None, None))



class BlockEvaluate_DataPreprocess(BlockEvaluate_Standard):
    '''
    In BlockEvaluate_Standard.evaluate() we only evaluate on the training_datapair.
    But here we want to evaluate both training and validation.
    The process flow will be almost identical otherwise.
    '''
    def __init__(self):
        super().__init__()
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_DataPreprocess Class" % (None, None))
        
        
    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def,#: BlockDefinition, 
                 training_datapair: ezDataSet,
                 validation_datapair: ezDataSet):
        ezLogging.info("%s - Start evaluating..." % (block_material.id))
        
        training_datapair = super().evaluate(block_material,
                                             block_def,
                                             training_datapair)
        validation_datapair = super().evaluate(block_material,
                                               block_def,
                                               validation_datapair)
        
        output = []
        if not block_material.dead:
            # NOTE the output of the BlockEvaluate_Standard.evaluate() should be a list of one element
            assert(len(training_datapair)==1)
            assert(len(validation_datapair)==1)
            output.append(training_datapair[0])
            output.append(validation_datapair[0])
        
        return output

