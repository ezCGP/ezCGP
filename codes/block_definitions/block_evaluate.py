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
from copy import deepcopy

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
    
    
    def standard_evaluate(self,
                          block_material: BlockMaterial,
                          block_def,#: BlockDefinition, 
                          input_dataset: ezDataSet):
        '''
        After a while of developing, we noticed that ALL our blocks followed the same eval process.
        the main difference was WHAT data was passed in...that's it!
        So this is going to be here (separate from 'evaluate()' as a quick-call method that can
        be used where needed.
        '''
        # add input data
        for i, data_input in enumerate(input_dataset):
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
                
        ezLogging.info("%s - Ending evaluating...%i output" % (block_material.id, len(output)))
        return output

    
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
    def train_graph(self):
        pass

    @abstractmethod
    def run_graph(self):
        pass
    
    def standard_build_graph(self,
                             block_material: BlockMaterial,
                             block_def,#: BlockDefinition, 
                             input_layers):
        '''
        trying to generalize the graph building process similar to standard_evaluate()
        '''
        # add input data
        for i, input_layer in enumerate(input_layer):
            block_material.evaluated[-1*(i+1)] = input_layer

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
                block_material.evaluated[node_index] = function(*inputs, *args)

        output = []
        if not block_material.dead:
            for output_index in range(block_def.main_count, block_def.main_count+block_def.output_count):
                output.append(block_material.evaluated[block_material.genome[output_index]])
                
        ezLogging.info("%s - Ending evaluating...%i output" % (block_material.id, len(output)))
        return output

    
    def preprocess_block_evaluate(self, block_material):
        '''
        should always happen before we evaluate...should be in BlockDefinition.evaluate()
        
        Note we can always customize this to our block needs which is why we included in BlockEvaluate instead of BlockDefinition
        '''
        super().preprocess_block_evaluate(block_material)
        block_material.graph = None


    def postprocess_block_evaluate(self, block_material):
        '''
        should always happen after we evaluate. important to blow away block_material.evaluated to clear up memory

        can always customize this method which is why we included it in BlockEvaluate and not BlockDefinition
        '''
        super().postprocess_block_evaluate(block_material)



class BlockEvaluate_Standard(BlockEvaluate_Abstract):
    '''
    This could be used for any basic application of methods onto data, like symbolic regression.
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_Standard Class" % (None, None))
        
        
    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def,#: BlockDefinition, 
                 training_datapair: ezDataSet):
        ezLogging.info("%s - Start evaluating..." % (block_material.id))
        output = self.standard_evaluate(block_material, block_def, training_datapair)
        block_material.output = output


    def preprocess_block_evaluate(self, block_material: BlockMaterial):
        '''
        we really could just remove this method...I think we're
        keeping it here as a reminder that we can change it
        '''
        super().preprocess_block_evaluate(block_material)



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
    
    
    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def, #: BlockDefinition,
                 training_datapair: ezDataSet,
                 validation_datapair: ezDataSet):
        ezLogging.info("%s - Start evaluating..." % (block_material.id))
        
        training_output = self.standard_evaluate(block_material, block_def, training_datapair)
        
        output = []
        if not block_material.dead:
            output.append(deepcopy(training_output[0])) #assuming only 1 output
            # here is the unique part...add in validation_datapair
            output.append(validation_datapair[0])
        
        block_material.output = output



class BlockEvaluate_TrainValidate(BlockEvaluate_Standard):
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
        
        # going to treat training + validation as separate block_materials!
        output = []
        for datapair in [training_datapair, validation_datapair]:
            single_output = self.standard_evaluate(block_material, block_def, datapair)
            if block_material.dead:
                return []
            else:
                output.append(deepcopy(single_output[0])) #assuming only one output
                self.preprocess_block_evaluate(block_material) #prep for next loop through datapair
        
        block_material.output = output



class BlockEvaluate_TFKerasGraph(BlockEvaluate_GraphAbstract):
    '''
    assuming block_def has these custom attributes:
     * num_classes
     * input_shape
    '''
    def __init__(self):
        super().__init__()
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_TFKerasGraph Class" % (None, None))
        
        
    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def,#: BlockDefinition, 
                 training_datapair: ezDataSet,
                 validation_datapair: ezDataSet):
        ''' stuff the old code has but unclear why
        gpus = tf.config.experimental.list_physical_devices('GPU')
        #tf.config.experimental.set_virtual_device_configuration(gpus[0],[
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 1024*3)
                ])
        '''
        import tensorflow as tf
        ezLogging.info("%s - Start evaluating..." % (block_material.id))
        # Assume input+output layers are going to be lists with only one element
        
        #https://www.tensorflow.org/api_docs/python/tf/keras/layers/InputLayer
        # vs
        #https://www.tensorflow.org/api_docs/python/tf/keras/Input
        input_layer = tf.keras.Input(input_shape=block_def.input_shape,
                                     batch_size=None,
                                     dtype=None)
        output_layer = self.standard_build_graph(block_material,
                                                  block_def,
                                                  [input_layer])[0]
        
        #  flatten the output node and perform a softmax
        output_flatten = tf.keras.layers.Flatten()(output_layer)
        logits = tf.keras.layers.Dense(units=block_def.num_classes, activation=None, use_bias=True)(output_flatten)
        softmax = tf.keras.layers.Softmax(logits, axis=1) # TODO verify axis...axis=1 was given by original code

        #https://www.tensorflow.org/api_docs/python/tf/keras/Model
        block_material.graph = tf.keras.Model(inputs=input_layer, outputs=softmax)
        
        #https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
        block_material.graph.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                     loss="categorical_crossentropy",
                                     metrics=None,
                                     loss_weights=None,
                                     weighted_metrics=None,
                                     run_eagerly=None)
        
        #https://www.tensorflow.org/api_docs/python/tf/keras/Model#train_on_batch
        train_batch_count = len(training_datapair.x_train) // block_def.batch_size
        validate_batch_count = len(validate_datapair.x_train) // block_def.batch_size
        for ith_epoch in range(block_def.epochs):
            
            train_batch_loss = 0
            for ith_batch in range(train_batch_count):
                input_batch, y_batch = training_datapair.next_batch(block_def.batch_size) #next_batch_train()
                train_batch_loss += block_material.graph.train_on_batch(x=input_batch, y=y_batch)
            train_batch_loss /= train_batch_count
            
            if i % 5 == 0:
                # get validation score
                validate_batch_loss = 0
                for ith_batch in range(train_batch_count):
                    input_batch, y_batch = validate_datapair.next_batch(block_def.batch_size) #next_batch_train()
                    validate_batch_loss += block_material.graph.test_on_batch(x=input_batch, y=y_batch)
                validate_batch_loss /= train_batch_count
                
                # TODO get accuracy metrics
                
        tf.keras.backend.clear_session()
        output = ... # validation metrics


