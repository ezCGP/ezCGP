'''

'''

### packages
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
from codes.block_definitions.evaluate.block_evaluate import BlockEvaluate_Abstract
from codes.utilities.custom_logging import ezLogging



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

        ezLogging.info("%s - Ending build graph...%i output" % (block_material.id, len(output)))
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

        input_layer = tf.keras.Input(shape=datapair.pipeline_wrapper.image_shape, dtype=None)
        output_layer = self.standard_build_graph(block_material,
                                                 block_def,
                                                 [input_layer])[0]

        #  flatten the output node and perform a softmax
        output_flatten = tf.keras.layers.Flatten()(output_layer)
        logits = tf.keras.layers.Dense(units=datapair.pipeline_wrapper.num_classes, activation=None, use_bias=True)(output_flatten)
        softmax = tf.keras.layers.Softmax(axis=1)(logits) # TODO verify axis...axis=1 was given by original code

        #https://www.tensorflow.org/api_docs/python/tf/keras/Model
        block_material.graph = tf.keras.Model(inputs=input_layer, outputs=softmax)

        #https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
        block_material.graph.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                     loss="categorical_crossentropy",
                                     metrics=[tf.keras.metrics.CategoricalAccuracy(),
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

        if training_datapair.images_wrapper.x is None:
            '''
            Here we assume that all our images are in directories that were fed directly into Augmentor.Pipeline at init
            so that we don't have to read in all the images at once before we batch them out.
            This means we can use the Augmentor.Pipeline.keras_generator() method
            https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.keras_generator

            NOT YET TESTED
            '''
            training_generator = training_datapair.pipeline_wrapper.keras_generator(batch_size=block_def.batch_size,
                                                scaled=True, #if errors, try setting to False
                                                image_data_format="channels_last", #or "channels_last"
                                               )
            validation_generator = validation_datapair.pipeline_wrapper.keras_generator(batch_size=block_def.batch_size,
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
                        preprocessing_function=training_datapair.pipeline_wrapper.pipeline.keras_preprocess_func()
                        )
            #training_datagen.fit(training_datapair.x) # don't need to call fit(); see documentation
            training_generator = training_datagen.flow(x=training_datapair.images_wrapper.x,
                                   y=training_datapair.images_wrapper.y,
                                   batch_size=block_def.batch_size,
                                   shuffle=True)

            validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                        preprocessing_function=validation_datapair.pipeline_wrapper.pipeline.keras_preprocess_func()
                        )
            #validation_datagen.fit(validation_datapair.x) # don't need to call fit(); see documentation
            validation_generator = training_datagen.flow(x=validation_datapair.images_wrapper.x,
                                 y=validation_datapair.images_wrapper.y,
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
                                                  training_datapair.images_wrapper.num_images//block_def.batch_size,
                                                  block_def.epochs))

        history = block_material.graph.fit(x=training_generator,
                           epochs=block_def.epochs,
                           verbose=2,
                           callbacks=None,
                           validation_data=validation_generator,
                           shuffle=True,
                           steps_per_epoch=training_datapair.images_wrapper.num_images//block_def.batch_size,
                           validation_steps=validation_datapair.images_wrapper.num_images//block_def.batch_size,
                           max_queue_size=10,
                           workers=1,
                           use_multiprocessing=False,
                          )
        tf.keras.backend.clear_session()
        return [-1 * history.history['val_categorical_accuracy'][-1], #mult by -1 since we want to maximize accuracy but universe optimization is minimization of fitness
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

        block_material.output = [None, output]



class BlockEvaluate_TFKeras_TransferLearning(BlockEvaluate_GraphAbstract):
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
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_TFKeras_TransferLearning Class" % (None, None))


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
                      training_datapair, # called datapair but really only the pipeline
                      validation_datapair):
        if training_datapair.images_wrapper.x is None:
            '''
            Here we assume that all our images are in directories that were fed directly into Augmentor.Pipeline at init
            so that we don't have to read in all the images at once before we batch them out.
            This means we can use the Augmentor.Pipeline.keras_generator() method
            https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.keras_generator

            NOT YET TESTED
            '''
            training_generator = training_datapair.pipeline_wrapper.pipeline.keras_generator(batch_size=block_def.batch_size,
                                                                                             scaled=True, #if errors, try setting to False
                                                                                             image_data_format="channels_last", #or "channels_last"
                                                                                            )
            validation_generator = validation_datapair.pipeline_wrapper.pipeline.keras_generator(batch_size=block_def.batch_size,
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
                                        preprocessing_function=training_datapair.pipeline_wrapper.pipeline.keras_preprocess_func()
                                        )
            #training_datagen.fit(training_datapair.x) # don't need to call fit(); see documentation
            training_generator = training_datagen.flow(x=training_datapair.images_wrapper.x,
                                                       y=training_datapair.images_wrapper.y,
                                                       batch_size=block_def.batch_size,
                                                       shuffle=True)

            validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                        preprocessing_function=validation_datapair.pipeline_wrapper.pipeline.keras_preprocess_func()
                                        )
            #validation_datagen.fit(validation_datapair.x) # don't need to call fit(); see documentation
            validation_generator = training_datagen.flow(x=validation_datapair.images_wrapper.x,
                                                         y=validation_datapair.images_wrapper.y,
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
                                                                                      training_datapair.images_wrapper.num_images//block_def.batch_size,
                                                                                      block_def.epochs))

        history = block_material.graph.fit(x=training_generator,
                                           epochs=block_def.epochs,
                                           verbose=2, # TODO set to 0 or 2 after done debugging
                                           callbacks=None,
                                           validation_data=validation_generator,
                                           shuffle=True,
                                           steps_per_epoch=training_datapair.images_wrapper.num_images//block_def.batch_size, # TODO
                                           validation_steps=validation_datapair.images_wrapper.num_images//block_def.batch_size,
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
            self.build_graph(block_material, block_def, training_datapair.pipeline_wrapper)
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

        block_material.output = [None, output]


