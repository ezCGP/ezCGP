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
import importlib
from torch import nn

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from data.data_tools.ezData import ezData
from data.data_tools.simganData import SimGANDataset
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
                 training_datapair: ezData,
                 validation_datapair: ezData=None):
        pass
    
    
    def standard_evaluate(self,
                          block_material,
                          block_def, 
                          input_list):
        '''
        After a while of developing, we noticed that ALL our blocks followed the same eval process.
        the main difference was WHAT data was passed in...that's it!
        So this is going to be here (separate from 'evaluate()' as a quick-call method that can
        be used where needed.
        '''
        # verify that the input data matches the expected datatypes
        for input_dtype, input_data in zip(block_def.input_dtypes, input_list):
            if input_dtype != type(input_data):
                ezLogging.critical("%s - Input data type (%s) doesn't match expected type (%s)" % (block_material.id, type(input_data), input_dtype))
                return None

        # add input data
        for i, data_input in enumerate(input_list):
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
                
        ezLogging.info("%s - Ending evaluating...%i output" % (block_material.id, len(output_list)))
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
    def evaluate(self):
        pass
    
    def standard_build_graph(self,
                             block_material: BlockMaterial,
                             block_def,#: BlockDefinition, 
                             input_layers = None):
        '''
        trying to generalize the graph building process similar to standard_evaluate()

        For Transfer Learning:
        we expect input_layers to be None, and later when we call function(*inputs, *args), we want to pass in
        an empty list for inputs.
        This also guarentees that no matter how many 'active nodes' we have for our transfer learning block,
        we will only ever use one pretrained model...no inputs are shared between nodes so the models never connect!
        '''
        # add input data
        if input_layers is not None:
            for i, input_layer in enumerate(input_layers):
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
                if input_layers is not None:
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
        block_material.graph = None



class BlockEvaluate_Standard(BlockEvaluate_Abstract):
    '''
    This could be used for any basic application of methods onto data, like symbolic regression.
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_Standard Class" % (None, None))
        
        
    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def,#: BlockDefinition, 
                 training_datapair: ezData,
                 validation_datapair: ezData=None):
        ezLogging.info("%s - Start evaluating..." % (block_material.id))
        output_list = self.standard_evaluate(block_material, block_def, training_datapair.x)
        #training_datapair.x = output_list[0]
        #block_material.output = training_datapair
        block_material.output = output_list


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
                 training_datapair: ezData,
                 validation_datapair: ezData):
        ezLogging.info("%s - Start evaluating..." % (block_material.id))
        
        output_list = self.standard_evaluate(block_material, block_def, [training_datapair.pipeline])
        training_datapair.pipeline = output_list[0] #assuming only outputs the pipeline
        
        block_material.output = [training_datapair, validation_datapair]



class BlockEvaluate_TrainValidate(BlockEvaluate_Standard):
    '''
    In BlockEvaluate_Standard.evaluate() we only evaluate on the training_datapair.
    But here we want to evaluate both training and validation.
    The process flow will be almost identical otherwise.
    '''
    def __init__(self):
        super().__init__()
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_TrainValidate Class" % (None, None))
        
        
    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def,#: BlockDefinition, 
                 training_datapair: ezData,
                 validation_datapair: ezData):
        ezLogging.info("%s - Start evaluating..." % (block_material.id))
        
        # going to treat training + validation as separate block_materials!
        output = []
        for datapair in [training_datapair, validation_datapair]:
            single_output_list = self.standard_evaluate(block_material, block_def, [datapair.pipeline])
            datapair.pipeline = single_output_list[0]
            if block_material.dead:
                return []
            else:
                output.append(datapair)
                self.preprocess_block_evaluate(block_material) #prep for next loop through datapair
 
        block_material.output = output



class BlockEvaluate_TFKeras(BlockEvaluate_GraphAbstract):
    '''
    assuming block_def has these custom attributes:
     * num_classes
     * input_shape
    '''
    def __init__(self):
        super().__init__()
        globals()['tf'] = importlib.import_module('tensorflow')
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_TFKeras Class" % (None, None))


    def build_graph(self, block_material, block_def, datapair):
        '''
        Assume input+output layers are going to be lists with only one element
        
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/InputLayer
         vs
        https://www.tensorflow.org/api_docs/python/tf/keras/Input
        '''
        ezLogging.debug("%s - Building Graph" % (block_material.id))

        input_layer = tf.keras.Input(shape=datapair.image_shape,
                                     batch_size=block_def.batch_size,
                                     dtype=None)
        output_layer = self.standard_build_graph(block_material,
                                                  block_def,
                                                  [input_layer])[0]

        #  flatten the output node and perform a softmax
        output_flatten = tf.keras.layers.Flatten()(output_layer)
        logits = tf.keras.layers.Dense(units=datapair.num_classes, activation=None, use_bias=True)(output_flatten)
        softmax = tf.keras.layers.Softmax(axis=1)(logits) # TODO verify axis...axis=1 was given by original code

        #https://www.tensorflow.org/api_docs/python/tf/keras/Model
        block_material.graph = tf.keras.Model(inputs=input_layer, outputs=softmax)
        
        #https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
        block_material.graph.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                     loss="categorical_crossentropy",
                                     metrics=[tf.keras.metrics.Accuracy(),
                                              tf.keras.metrics.Precision(),
                                              tf.keras.metrics.Recall()],
                                     loss_weights=None,
                                     weighted_metrics=None,
                                     run_eagerly=None)


    def old_train_graph(self,
                    block_material,
                    block_def,
                    training_datapair,
                    validation_datapair):
        '''
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#train_on_batch
        '''
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
        output = ting # validation metrics
        return output
        

    def get_generator(self,
                      block_material,
                      block_def,
                      training_datapair,
                      validation_datapair):
        
        if training_datapair.x is None:
            '''
            Here we assume that all our images are in directories that were fed directly into Augmentor.Pipeline at init
            so that we don't have to read in all the images at once before we batch them out.
            This means we can use the Augmentor.Pipeline.keras_generator() method
            https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.keras_generator

            NOT YET TESTED
            '''
            training_generator = training_datapair.keras_generator(batch_size=block_def.batch_size,
                                                                   scaled=True, #if errors, try setting to False
                                                                   image_data_format="channels_last", #or "channels_last"
                                                                  )
            validation_generator = validation_datapair.keras_generator(batch_size=block_def.batch_size,
                                                                       scaled=True, #if errors, try setting to False
                                                                       image_data_format="channels_last", #or "channels_last"
                                                                      )
        else:
            '''
            Here we assume that we have to load all the data into datapair.x and .y so we have to pass the
            Augmentor.Pipeline as a method fed into tf.keras.preprocessing.image.ImadeDataGenerator
            https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.keras_preprocess_func
            https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
            '''
            training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                        preprocessing_function=training_datapair.pipeline.keras_preprocess_func()
                                        )
            #training_datagen.fit(training_datapair.x) # don't need to call fit(); see documentation
            training_generator = training_datagen.flow(x=training_datapair.x,
                                                       y=training_datapair.y,
                                                       batch_size=block_def.batch_size,
                                                       shuffle=True)

            validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                        preprocessing_function=validation_datapair.pipeline.keras_preprocess_func()
                                        )
            #validation_datagen.fit(validation_datapair.x) # don't need to call fit(); see documentation
            validation_generator = training_datagen.flow(x=validation_datapair.x,
                                                         y=validation_datapair.y,
                                                         batch_size=block_def.batch_size,
                                                         shuffle=True)
            
        return training_generator, validation_generator
    
    
    def train_graph(self,
                    block_material,
                    block_def,
                    training_datapair,
                    validation_datapair):
        ezLogging.debug("%s - Building Generators" % (block_material.id))
        training_generator, validation_generator = self.get_generator(block_material,
                                                                      block_def,
                                                                      training_datapair,
                                                                      validation_datapair)

        ezLogging.debug("%s - Training Graph - %i batch size, %i steps, %i epochs" % (block_material.id,
                                                                                      block_def.batch_size,
                                                                                      training_datapair.num_images//block_def.batch_size,
                                                                                      block_def.epochs))

        history = block_material.graph.fit(x=training_generator,
                                           epochs=block_def.epochs,
                                           verbose=2, # TODO set to 0 after done debugging
                                           callbacks=None,
                                           validation_data=validation_generator,
                                           shuffle=True,
                                           steps_per_epoch=training_datapair.num_images//block_def.batch_size, # TODO
                                           validation_steps=validation_datapair.num_images//block_def.batch_size,
                                           max_queue_size=10,
                                           workers=1,
                                           use_multiprocessing=False,
                                          )
        tf.keras.backend.clear_session()
        output = history.stuff # validation metrics
        return [-1 * history.history['val_accuracy'][-1], #mult by -1 since we want to maximize accuracy but universe optimization is minimization of fitness
                -1 * history.history['val_precision'][-1],
                -1 * history.history['val_recall'][-1]]

        
    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def,#: BlockDefinition, 
                 training_datapair: ezData,
                 validation_datapair: ezData):
        '''
        stuff the old code has but unclear why
        
            gpus = tf.config.experimental.list_physical_devices('GPU')
            #tf.config.experimental.set_virtual_device_configuration(gpus[0],[
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 1024*3)
                    ])
        '''
        ezLogging.info("%s - Start evaluating..." % (block_material.id))
        try:
            self.build_graph(block_material, block_def, training_datapair)
        except Exception as err:
            ezLogging.critical("%s - Build Graph; Failed: %s" % (block_material.id, err))
            block_material.dead = True
            import pdb; pdb.set_trace()
            return

        try:
            output = self.train_graph(block_material, block_def, training_datapair, validation_datapair)
        except Exception as err:
            ezLogging.critical("%s - Train Graph; Failed: %s" % (block_material.id, err))
            block_material.dead = True
            import pdb; pdb.set_trace()
            return
        
        block_material.output = output # TODO make sure it is a list



class BlockEvaluate_TFKeras_TransferLearning(BlockEvaluate_GraphAbstract):
    '''
    Here we will initialize our tf.keras.Model with a pretrained network.
    We expect another TFKeras Block to finish the Model and compile it then.
    So this block does not handle compiling or training.

    TODO - consider setting pretrained network layers to 'untrainable'
    '''
    def __init__(self):
        super().__init__()
        globals()['tf'] = importlib.import_module('tensorflow')
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_TFKeras_TransferLearning Class" % (None, None))


    def build_graph(self, block_material, block_def, datapair):
        ezLogging.debug("%s - Building Graph" % (block_material.id))

        input_layer, output_layer = self.standard_build_graph(block_material,
                                                              block_def)[0]
        '''
        TODO don't use standard build graph but write new method that grabs last active node
        and only uses that to start the graph
        '''

        # attach layers to datapair so it is available to the next block
        datapair.graph_input_layer = input_layer
        datapair.final_pretrained_layer = output_layer


    def train_graph(self):
        pass

        
    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def,#: BlockDefinition, 
                 training_datapair: ezData,
                 validation_datapair: ezData):
        ezLogging.info("%s - Start evaluating..." % (block_material.id))

        try:
            self.build_graph(block_material, block_def, training_datapair)
        except Exception as err:
            ezLogging.critical("%s - Build Graph; Failed: %s" % (block_material.id, err))
            block_material.dead = True
            import pdb; pdb.set_trace()
            return

        block_material.output = [training_datapair, validation_datapair]



class BlockEvaluate_TFKeras_TransferLearning2(BlockEvaluate_GraphAbstract):
    '''
    Here we will initialize our tf.keras.Model with a pretrained network.
    We expect another TFKeras Block to finish the Model and compile it then.
    So this block does not handle compiling or training.

    Unlike the original BlockEvaluate_TFKeras_TransferLearning() there are several
    assumptions:
     * main_nodes count is 1
     * if no main_node is active, kill the individual
     * create an input layer so we can apply preprocess_input() method as layer

    TODO - consider setting pretrained network layers to 'untrainable'
    '''
    class TooManyMainNodes(Exception):
        '''
        Our assumption is that this block can only have 1 main node
        '''
        def __init__(self, main_count):
            self.main_count = main_count
            self.message = "An assumption in this block is that it can only have 1 main node, not %i" % main_count
            super().__init__(self.message)


    class NoActiveMainNodes(Exception):
        '''
        Our assumption is that this block can only have 1 main node, and any individual
        has to have that node active or else kill it.
        '''
        def __init__(self, active_nodes):
            self.active_nodes = active_nodes
            self.message = "The 0th and only main node is not active: %s" % active_nodes
            super().__init__(self.message)


    def __init__(self):
        super().__init__()
        globals()['tf'] = importlib.import_module('tensorflow')
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_TFKeras_TransferLearning2 Class" % (None, None))


    def build_graph(self, block_material, block_def, datapair):
        ezLogging.debug("%s - Building Graph" % (block_material.id))


        input_layer = tf.keras.layers.Input(shape=datapair.image_shape)
        # doc for preprocess_input says it expects floating point np array or tensor unless I misunderstood
        next_layer = tf.cast(input_layer, tf.float32)

        # now grab 0th and only active main node
        function = block_material[0]["ftn"]
        inputs = [next_layer]
        args = []
        for node_arg_index in block_material[0]["args"]:
            args.append(block_material.args[node_arg_index].value)

        ezLogging.debug("%s - Eval %i; Function: %s, Inputs: %s, Args: %s" % (block_material.id, 0, function, inputs, args))
        output_layer = function(*inputs, *args)

        ezLogging.info("%s - Ending evaluating...%i output" % (block_material.id, 1))
        # attach layers to datapair so it is available to the next block
        datapair.graph_input_layer = input_layer
        datapair.final_pretrained_layer = output_layer


    def train_graph(self):
        pass

        
    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def,#: BlockDefinition, 
                 training_datapair: ezData,
                 validation_datapair: ezData):
        ezLogging.info("%s - Start evaluating..." % (block_material.id))

        try:
            # check if this fails our conditions
            if block_def.main_count != 1:
                raise self.TooManyMainNodes(block_def.main_count)
            if 0 not in block_material.active_nodes:
                raise self.NoActiveMainNodes(block_material.active_nodes)
            # good to continue to build
            self.build_graph(block_material, block_def, training_datapair)
        except Exception as err:
            ezLogging.critical("%s - Build Graph; Failed: %s" % (block_material.id, err))
            block_material.dead = True
            import pdb; pdb.set_trace()
            return

        block_material.output = [training_datapair, validation_datapair]



class BlockEvaluate_TFKeras_AfterTransferLearning(BlockEvaluate_GraphAbstract):
    '''
    Should follow a BlockEvaluate_TFKeras_TransferLearning Block so that it's input is the final layer
    of the pretrained tf.keras model.
    In this block, we'll finish building the graph, compile and train.
    '''
    def __init__(self):
        super().__init__()
        globals()['tf'] = importlib.import_module('tensorflow')
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_TFKeras_AfterTransferLearning Class" % (None, None))


    def build_graph(self, block_material, block_def, datapair):
        '''
        Assume input+output layers are going to be lists with only one element
        
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/InputLayer
         vs
        https://www.tensorflow.org/api_docs/python/tf/keras/Input
        '''
        ezLogging.debug("%s - Building Graph" % (block_material.id))

        output_layer = self.standard_build_graph(block_material,
                                                  block_def,
                                                  [datapair.final_pretrained_layer])[0]

        #  flatten the output node and perform a softmax
        output_flatten = tf.keras.layers.Flatten()(output_layer)
        logits = tf.keras.layers.Dense(units=datapair.num_classes, activation=None, use_bias=True)(output_flatten)
        softmax = tf.keras.layers.Softmax(axis=1)(logits) # TODO verify axis...axis=1 was given by original code

        #https://www.tensorflow.org/api_docs/python/tf/keras/Model
        block_material.graph = tf.keras.Model(inputs=datapair.graph_input_layer, outputs=softmax)
        
        #https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
        block_material.graph.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                     loss="categorical_crossentropy",
                                     metrics=[tf.keras.metrics.Accuracy(),
                                              tf.keras.metrics.Precision(),
                                              tf.keras.metrics.Recall()],
                                     loss_weights=None,
                                     weighted_metrics=None,
                                     run_eagerly=None)
        

    def get_generator(self,
                      block_material,
                      block_def,
                      training_datapair,
                      validation_datapair):
        
        if training_datapair.x is None:
            '''
            Here we assume that all our images are in directories that were fed directly into Augmentor.Pipeline at init
            so that we don't have to read in all the images at once before we batch them out.
            This means we can use the Augmentor.Pipeline.keras_generator() method
            https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.keras_generator

            NOT YET TESTED
            '''
            training_generator = training_datapair.keras_generator(batch_size=block_def.batch_size,
                                                                   scaled=True, #if errors, try setting to False
                                                                   image_data_format="channels_last", #or "channels_last"
                                                                  )
            validation_generator = validation_datapair.keras_generator(batch_size=block_def.batch_size,
                                                                       scaled=True, #if errors, try setting to False
                                                                       image_data_format="channels_last", #or "channels_last"
                                                                      )
        else:
            '''
            Here we assume that we have to load all the data into datapair.x and .y so we have to pass the
            Augmentor.Pipeline as a method fed into tf.keras.preprocessing.image.ImadeDataGenerator
            https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.keras_preprocess_func
            https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

            keras_preprocess_func doc:
            "The function will run after the image is resized and augmented. The function should take one
            argument: one image (Numpy tensor with rank 3), and should output a Numpy tensor with the same shape."
            So we can't have any Augmentor.Operations that change the shape of the data or else we'll get an
            error like: 'ValueError: could not broadcast input array from shape (...) into shape (...)'
            '''
            training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                        preprocessing_function=training_datapair.pipeline.keras_preprocess_func()
                                        )
            #training_datagen.fit(training_datapair.x) # don't need to call fit(); see documentation
            training_generator = training_datagen.flow(x=training_datapair.x,
                                                       y=training_datapair.y,
                                                       batch_size=block_def.batch_size,
                                                       shuffle=True)

            validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                        preprocessing_function=validation_datapair.pipeline.keras_preprocess_func()
                                        )
            #validation_datagen.fit(validation_datapair.x) # don't need to call fit(); see documentation
            validation_generator = training_datagen.flow(x=validation_datapair.x,
                                                         y=validation_datapair.y,
                                                         batch_size=block_def.batch_size,
                                                         shuffle=True)
            
        return training_generator, validation_generator
    
    
    def train_graph(self,
                    block_material,
                    block_def,
                    training_datapair,
                    validation_datapair):
        ezLogging.debug("%s - Building Generators" % (block_material.id))
        training_generator, validation_generator = self.get_generator(block_material,
                                                                      block_def,
                                                                      training_datapair,
                                                                      validation_datapair)

        ezLogging.debug("%s - Training Graph - %i batch size, %i steps, %i epochs" % (block_material.id,
                                                                                      block_def.batch_size,
                                                                                      training_datapair.num_images//block_def.batch_size,
                                                                                      block_def.epochs))
        '''
        for i, data in enumerate(training_generator):
            print(i, data[0].shape, data[1].shape)
            # 0 (5, 32, 32, 3) (5, 10)
            if i == 31:
                # why did I do this?
                import pdb; pdb.set_trace()'''
        
        history = block_material.graph.fit(x=training_generator,
                                           epochs=block_def.epochs,
                                           verbose=2, # TODO set to 0 or 2 after done debugging
                                           callbacks=None,
                                           validation_data=validation_generator,
                                           shuffle=True,
                                           steps_per_epoch=training_datapair.num_images//block_def.batch_size, # TODO
                                           validation_steps=validation_datapair.num_images//block_def.batch_size,
                                           max_queue_size=10,
                                           workers=1,
                                           use_multiprocessing=False,
                                          )
        tf.keras.backend.clear_session()

        #output = history.stuff # validation metrics
        # NOTE: this is essentially our individual.fitness.values
        return [-1*history.history['val_accuracy'][-1], -1*history.history['val_precision'][-1], -1*history.history['val_recall'][-1]]
        #return [-1*history.history['val_precision'][-1], -1*history.history['val_recall'][-1]]

        
    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def,#: BlockDefinition, 
                 training_datapair: ezData,
                 validation_datapair: ezData):
        ezLogging.info("%s - Start evaluating..." % (block_material.id))
        try:
            self.build_graph(block_material, block_def, training_datapair)
        except Exception as err:
            ezLogging.critical("%s - Build Graph; Failed: %s" % (block_material.id, err))
            block_material.dead = True
            import pdb; pdb.set_trace()
            return

        try:
            # outputs a list of the validation metrics
            output = self.train_graph(block_material, block_def, training_datapair, validation_datapair)
        except Exception as err:
            ezLogging.critical("%s - Train Graph; Failed: %s" % (block_material.id, err))
            block_material.dead = True
            import pdb; pdb.set_trace()
            return
        
        block_material.output = [None, output] # TODO make sure it is a list

class BlockEvaluate_PyTorch_Abstract(BlockEvaluate_GraphAbstract):
    """
    An abstract class defining the structure of a PyTorch neural network BlockEvaluate. Provides a function to build a PyTorch nueral network graph
    from PyTorchLayerWrapper operators.
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def build_graph(self):
        pass

    @abstractmethod
    def train_graph(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def standard_build_graph(self, block_material, block_def, input_list):
        """
        Builds a PyTorch nueral network graph from PyTorchLayerWrapper operators. Because many PyTorch layers require you know information about
        the shape of the input, PyTorchLayerWrappers are used to calculate output shapes and string together the PyTorch graph in this function
        
        Parameters:
            block_material (BlockMaterial): an object containing the genetic material of the block
            block_def (BlockDefinition): an object holding the functions of the block
            input_list (list): list of inputs, each input should have a shape attribute
        
        Returns:
            graph (torch.nn.Sequential): the built neural network up 

        """
        from codes.block_definitions.utilities.operators_pytorch import InputLayer

        # add input data
        for i, input in enumerate(input_list):
            input_layer = InputLayer(input)
            block_material.evaluated[-1*(i+1)] = input_layer

        layers = []
        for node_index in block_material.active_nodes:
            if node_index < 0:
                # do nothing. at input node
                continue
            elif node_index >= block_def.main_count:
                # do nothing.
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

                ezLogging.debug("%s - Builing %i; Function: %s, Inputs: %s, Args: %s" % (block_material.id, node_index, function, inputs, args))
                node = function(*inputs, *args)
                block_material.evaluated[node_index] = node
                layers.append(node.get_layer())

        output_list = []
        if not block_material.dead:
            for output_index in range(block_def.main_count, block_def.main_count+block_def.output_count):
                output_list.append(block_material.evaluated[block_material.genome[output_index]])

        return nn.Sequential(*layers), output_list


class BlockEvaluate_SimGAN_Refiner(BlockEvaluate_PyTorch_Abstract):
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_SimGAN_Refiner Class" % (None, None))

    def build_graph(self, block_material, block_def, data):
        '''
        Build the SimGAN refiner network
        '''
        ezLogging.debug("%s - Building Graph" % (block_material.id))

        graph, output_list = self.standard_build_graph(block_material, block_def, [data.real_raw[0], data.real_raw[0]]) # We don't need all the data, just a single row to get the shape of the input
        output_shape = output_list[0].get_out_shape()
        ezLogging.info("%s - Ending building..." % (block_material.id))
        """
        Per my conversation with Rodd:
        - The way this is set up, we would only be able to have a single linear input
        - We want to be able to concat other inputs in
        - We can do this by changing the inputs list to be able to receive multiple inputs (main_input, misc_input1, misc_input2, ...)
        - For most layers, we won't need to touch any input besides main_input (which is data that is being directly acted upon by the layers)
        - We can make specialized layers that incorporate other misc_inputs into main_input
        """

        # Add a layer to get the output back into the right shape
        # TODO: consider using a linear layer instead of a conv1d and then unflattening. Not sure what is the right move
        extra_layers = []
        if len(output_shape) == 1:
            extra_layers.append(nn.Unflatten(dim=0, unflattened_size=(1,)))
        # Tanh to keep our output in the range of 0 to 1
        extra_layers = extra_layers + [nn.Conv1d(output_shape[0], 1, kernel_size=1), nn.Tanh()]

        block_material.graph = nn.Sequential(graph, *extra_layers)

    def train_graph(self,
                    block_material,
                    block_def,
                    training_datapair,
                    validation_datapair):
        '''
        The graph will be passed to the discriminator and trained there, so we pass here.
        '''
        pass
       
    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def,#: BlockDefinition,
                 train_data: SimGANDataset,
                 validation_data: SimGANDataset):
        '''
        All we have to do in evaluate is build the graph.
        '''
        ezLogging.info("%s - Start evaluating..." % (block_material.id))

        try:
            self.build_graph(block_material, block_def, train_data)
        except Exception as err:
            ezLogging.critical("%s - Build Graph; Failed: %s" % (block_material.id, err))
            block_material.dead = True
            import pdb; pdb.set_trace()
            return

        block_material.output = [train_data, validation_data, block_material.graph]


class BlockEvaluate_SimGAN_Discriminator(BlockEvaluate_PyTorch_Abstract):
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_SimGAN_Discriminator Class" % (None, None))
        # TODO: add implementation

    def build_graph(self, block_material, block_def, data):
        '''
        Build the SimGAN discriminator network
        '''
        ezLogging.debug("%s - Building Graph" % (block_material.id))

        graph, output_list = self.standard_build_graph(block_material, block_def, [data.real_raw[0], data.real_raw[0]]) # We don't need all the data, just a single row to get the shape of the input
        output_shape = output_list[0].get_out_shape()
        ezLogging.info("%s - Ending building..." % (block_material.id))

        # Add a final linear layer and get the output in shape Nx2
        in_features = 1
        for dim in list(output_shape): 
            in_features *= dim
        extra_layers = [nn.Linear(in_features, 2), nn.Flatten(start_dim=1)]

        block_material.graph = nn.Sequential(graph, *extra_layers)

    def train_graph(self,
                    block_material,
                    block_def,
                    training_datapair,
                    validation_datapair,
                    refiner):
        '''
        TODO: implement and add documentation 
        '''
        ezLogging.debug("%s - Training Graph - %i batch size, %i steps, %i epochs" % (block_material.id,
                                                                                      block_def.batch_size,
                                                                                      training_datapair.num_images//block_def.batch_size,
                                                                                      block_def.epochs))

    def evaluate(self,
                 block_material: BlockMaterial,
                 block_def,#: BlockDefinition,
                 train_data: SimGANDataset,
                 validation_data: SimGANDataset):
        '''
        TODO: implement and add documentation 
        '''
        ezLogging.info("%s - Start evaluating..." % (block_material.id))
        # import pdb; pdb.set_trace()
        train_data, refiner = train_data # we pack refiner in, hacky but convenient, we unpack it here
        try:
            self.build_graph(block_material, block_def, train_data)
        except Exception as err:
            ezLogging.critical("%s - Build Graph; Failed: %s" % (block_material.id, err))
            block_material.dead = True
            import pdb; pdb.set_trace()
            return
        
        import pdb; pdb.set_trace()
        output = self.train_graph(block_material, block_def, train_data, validation_data, refiner)
