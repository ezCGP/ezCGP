
# external packages
import numpy as np
import time
import shlex
import subprocess
import os
import tempfile
import gc

# my scripts
from mate_methods import Mate
from mutate_methods import Mutate
import tensorflow as tf

import operators
import arguments
import logging

import tensorflow as tf
from utils.DataSet import DataSet

class Block(Mate, Mutate):
    """
    Define the 'subclass' for a genome.
    Input nodes -> Block Instance -> Block Instance -> ... -> Output Nodes
    """

    def __init__(self, nickname,
                 setup_dict_ftn, setup_dict_arg, setup_dict_mate, setup_dict_mut,
                 operator_dict, block_input_dtypes, block_outputs_dtypes, block_main_count, block_arg_count,
                 block_mut_prob, block_mate_prob,
                 tensorblock_flag=False, learning_required=False, apply_to_val = True, num_classes=None, batch_size=None, n_epochs=1, large_dataset=None):
        tf.keras.Sequential
        tf.Session
        # TODO consider changing ftn_dict, arg_dict, etc to setup_dict_ftn, setup_dict_mate, etc
        # and then change gene_dict back to oper_dict or ftn_dict

        """
        take in a dictionary for each: key is the method, and value gives how to often that method will show
        values should go from (0,1]; anything geq to 1 means that we want this equally distributed with whatever is left.
        the sum of values less than 1 should not be greater than 1 or it will error
        """

        # inherit Mate, Mutate
        Mate.__init__(self, block_input_dtypes, block_outputs_dtypes,
                            block_main_count, block_arg_count)
        self.mate_prob = block_mate_prob
        self.mate_methods = list(setup_dict_mate.keys())
        self.mate_weights = self.buildWeights('mate_methods', setup_dict_mate)
        self.mate_dict = setup_dict_mate

        Mutate.__init__(self, block_input_dtypes, block_outputs_dtypes,
                              block_main_count, block_arg_count)
        self.mut_prob = block_mut_prob
        self.mut_methods = list(setup_dict_mut.keys())
        self.mut_weights = self.buildWeights('mut_methods', setup_dict_mut)
        self.mut_dict = setup_dict_mut

        #Fitness.__init__(self)

        # Block - MetaData
        self.name = nickname
        self.tensorblock_flag = tensorblock_flag
        self.apply_to_val = apply_to_val
        self.logs_path = tempfile.mkdtemp()
        self.learning_required = learning_required #
        self.num_classes = 10 # number of classes for what we are trying to classify...only relevant if there is supposed to be a learner
        self.need_evaluate = True # all new blocks need to be evaluated
        self.batch_size = batch_size # takes the batch_size from the block skeleton
        self.n_epochs = n_epochs # takes the n_epochs from the block skeleton
        self.large_dataset = large_dataset
        # Block - Argument List
        # the key is to make sure that every single individual in the population has an a list for self.args where each index/element is the same datatype
        self.arg_methods = list(setup_dict_arg.keys())
        self.arg_weights = self.buildWeights('arg_methods', setup_dict_arg)
        self.fillArgs()

        # Block - Genome List - Input Nodes
        #  should already be set by Genome()
        # Block - Genome List - Main Nodes
        self.ftn_methods = list(setup_dict_ftn.keys())
        self.ftn_weights = self.buildWeights('ftn_methods', setup_dict_ftn)
        self.operator_dict = operator_dict
        self.fillGenome()

        # Block - Evaluation
        self.resetEvalAttr()

        self.dataset = DataSet([], []) #this is for feeding the data in batches

    def __setitem__(self, node_index, dict_):
        # NOTE, isn't a dict if updating output node but shouldn't matter
        self.genome[node_index] = dict_


    def __getitem__(self, node_index):
        return self.genome[node_index]


    def mutate(self):
        roll = np.random.random()
        if roll <= self.mut_prob:
            # then mutate
            mutate_method = np.random.choice(a=self.mut_methods, size=1, p=self.mut_weights)[0]
            mutate_method(self, *self.mut_dict[mutate_method]['args'])
            # make sure in mutate_methods.py we have class Mutate() where we define all these methods
            self.need_evaluate = True
        else:
            # don't mutate
            pass


    def mate(self, other):
        roll = np.random.random()
        if roll <= self.mate_prob:
            # then mate
            mate_method = np.random.choice(a=self.mate_methods, size=1, p=self.mate_weights)[0]
            offspring_list = mate_method(self, other, *self.mate_dict[mate_method]['args'])
            for offspring in offspring_list:
                offspring.need_evaluate = True
                offspring.findActive() #will need active nodes for mutating step
            return offspring_list
        else:
            # don't mate
            return None

    def resetEvalAttr(self):
        self.dead = False
        self.evaluated = [None] * self.genome_count
        self.val_evaluated = [None] * self.genome_count
        self.genome_output_values = []
        self.validation_pair_output = None
        if self.tensorblock_flag:
            self.graph = tf.Graph()
            self.feed_dict = {}
            self.fetch_nodes = []
#            with self.graph.as_default():
 #               saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

        else:
            pass


    def tensorflow_preprocess(self, fetch_nodes, feed_dict, data_pair):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            tf_outputs = sess.run(
                fetches=fetch_nodes,
                feed_dict=feed_dict)
            val_outputs = None
            if self.apply_to_val:
                x_batch = tf.get_default_graph().get_operation_by_name('x_batch').outputs[0]
                feed_dict[x_batch] = data_pair["x_val"]
                val_outputs = sess.run(
                fetches=fetch_nodes,
                feed_dict=feed_dict)[0]
            return tf_outputs[0], val_outputs


    def tensorblock_evaluate(self, fetch_nodes, feed_dict, data_pair):
        '''
        fetch_nodes: output nodes
        feed_dict: block input
        returns finalOut, which is validation class predictions
        '''
        # large_dataset implies the dataset is too large to be completely stored
        # in memory and should be loaded from each of the files mentioned
        large_dataset = self.large_dataset
        try:
            if (large_dataset is None):
                fnames = ["not_large"]
            else:
                fnames, load_fname = large_dataset
            # initialize variables needed for batch feeding
            self.dataset = DataSet(data_pair['x_train'], data_pair['y_train'])
            x_val = data_pair["x_val"]
            y_val = data_pair["y_val"]
            with tf.Session(graph=self.graph) as sess:
                sess.run(tf.global_variables_initializer())

                # get placeholder nodes that will be fed to
                x_batch = tf.get_default_graph().get_operation_by_name('x_batch').outputs[0]
                y_batch = tf.get_default_graph().get_operation_by_name('y_batch').outputs[0]

                # n_epochs: number of epochs to run for while training
                # batch_size: size of the batch
                n_epochs = self.n_epochs
                batch_size = self.batch_size
                return_outputs = []
                # training on (X_train, y_train) dataset
                for epoch in range(n_epochs):
                    epoch_loss = 0
                    for fname in fnames:
                        if large_dataset:
                            X, y = load_fname(fname)
                            self.dataset = DataSet(X, y)
                        for step in range(int(np.ceil(self.dataset._num_examples/batch_size))):
                            X_train, y_train = self.dataset.next_batch(batch_size)
                            feed_dict[x_batch] = X_train
                            feed_dict[y_batch] = y_train
                            tf_outputs = sess.run(
                                fetches=fetch_nodes,
                                feed_dict=feed_dict)

                            tf_output_dict = tf_outputs[0]
                            step_loss = tf_output_dict['loss']
                            if large_dataset:
                                logging.info("epoch: {} loaded batch index: {}. Fed {}/{} samples. Step loss: {}"\
                                       .format(epoch, step, step * batch_size, self.dataset._num_examples, step_loss))
                            epoch_loss += step_loss

                        logging.info('Epoch completed. epoch_loss: {}'.format(epoch_loss))

                # Test on validation dataset from tester.py
                self.dataset = DataSet(x_val, y_val)
                batch_size = 256
                finalOut = []
                for step in range(int(np.ceil(self.dataset._num_examples/batch_size))):
                    X_train, y_train = self.dataset.next_batch(batch_size)
                    feed_dict[x_batch] = X_train
                    feed_dict[y_batch] = y_train

                    tf_outputs = sess.run(
                        fetches=fetch_nodes[:-2],
                        feed_dict=feed_dict)
                    tf_output_dict = tf_outputs[0]
                    outs = tf_output_dict["classes"].tolist()
                    finalOut += outs
                return np.array(finalOut)
        except ValueError as e:
            logging.info(e)
            logging.info ("Mismatched shapes of tensors leading to error at evaluation time. ")
            self.dead = True
            return tf_output_dict["classes"] # not really sure how to return properly after setting to dead... but this runs...

    def tensorboard_show(self, wait_seconds=60):
        # only should work after tensorblock_evaluate() is run
        cmd = "tensorboard --log_dir=%s" % self.logs_path
        args = shelx.split(cmd)
        p = subprocess.Popen(args)
        logging.info("tensorboard link created for %iseconds" % wait_seconds)
        time.sleep(wait_seconds)
        p.terminate() # or p.kill()
        logging.info("tensorbard killed")

    def tensorflow_add_optimizer_loss_layer(self):
        for output_node in range(self.genome_main_count, self.genome_main_count+self.genome_output_count):
            flattened = tf.layers.Flatten()(self.evaluated[self[output_node]])
            labels = tf.placeholder(tf.int32, [None], name='y_batch')
            logits = tf.layers.dense(inputs=flattened, units=self.num_classes) # logits layer
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                        logits = logits,
                                        labels = tf.one_hot(indices=labels, depth=self.num_classes, dtype=tf.float32)))
            tf.summary.tensor_summary("loss", loss)
            self.evaluated[output_node] = {
                "classes": tf.argmax(input=logits, axis=1),
                "probabilities": tf.nn.softmax(logits),
                "loss": loss}
            self.fetch_nodes.append(self.evaluated[output_node])
            tf.summary.tensor_summary("classes", self.evaluated[output_node]["classes"])
            tf.summary.tensor_summary("probabilities", self.evaluated[output_node]["probabilities"])
        optimizer = tf.train.AdamOptimizer() # TODO add optimizer into 'arguments' to and apply GA to it for mutate + mate
        step = tf.Variable(0, name='backprop_steps', trainable=False)
        train_step = optimizer.minimize(loss, global_step=step) #global_step)
        #tf.summary.scalar('loss', loss)
        #tf.summary.scalar('logits', logits)
        #tf.summary.scalar('results', results)
        merged_summary = tf.summary.merge_all()
        for graph_metadata in [train_step, merged_summary]: # opportunity to add other things we would want to fetch from the graph
            # remember, we need 'train_step' so that the optimizer is run; we don't actually need the output
            self.fetch_nodes.append(graph_metadata)

    def calculate_func_args_inputs(self):
        for node_index in self.active_nodes:
            if node_index < 0 or node_index >= self.genome_main_count:
                # nothing to evaluate at input/output nodes
                continue
            else:
                # calculate on main nodes
                pass
            # get functioin, inputs, arguments to evaluate
            function = self[node_index]["ftn"]
            inputs = []
            val_inputs = []
            node_input_indices = self[node_index]["inputs"]
            for node_input_index in node_input_indices:
                inputs.append(self.evaluated[node_input_index])
                val_inputs.append(self.val_evaluated[node_input_index])
            args = []
            node_arg_indices = self[node_index]["args"]
            for node_arg_index in node_arg_indices:
                args.append(self.args[node_arg_index].value)
            try:
                if self.tensorblock_flag:
                    if self.operator_dict[function]["include_labels"]:
                        raise(Exception("Tensorflow operators should not include labels"))
                    # added because the objects themselves were being sent in
                    argnums = [arg.value if type(arg) is not int \
                            else arg for arg in args]
                    with self.graph.as_default():
                        # building tensorflow graph
                        self.evaluated[node_index] = function(*inputs, *argnums)
                else:
                    if self.apply_to_val:
                        if self.operator_dict[function]["include_labels"]:
                            raise(Exception("Should not include labels in apply_to_val function")) #self.dead?
                        else:
                            self.val_evaluated[node_index] = function(*val_inputs, *args) #self.labels remains unchanged
                    if self.operator_dict[function]["include_labels"]:
                        self.evaluated[node_index], self.labels = function(*inputs, self.labels, *args)
                    else:
                        self.evaluated[node_index] = function(*inputs, *args)
            except Exception as e:
                raise(e)
                self.dead = True
                break

    def tensorflow_evaluate(self, block_inputs, labels_all, validation_pair):
        data_pair = {}
        for i, input_ in enumerate(block_inputs): # wont there only be one block input?
            logging.debug(i)
            data_dimension = list(input_.shape)
            data_dimension[0] = None # variable input size, "how to tell tensorflow" to figure it out by def.
            with self.graph.as_default():
                batch_X = tf.placeholder(tf.float32, data_dimension, name='x_batch')
                self.evaluated[-1*(i+1)] = batch_X
            self.feed_dict[batch_X] = input_
            data_pair['x_train'] = input_
            data_pair['y_train'] = labels_all
            data_pair["x_val"] = validation_pair[0]
            data_pair["y_val"] = validation_pair[1]
        self.calculate_func_args_inputs()
        if not self.dead:
            with self.graph.as_default():
                if self.learning_required:
                    self.tensorflow_add_optimizer_loss_layer()
                    try:
                        # now that the graph is built, we evaluate here
                        self.genome_output_values = [self.tensorblock_evaluate(self.fetch_nodes, self.feed_dict, data_pair), None]
                    except Exception as e:
                        raise(e)
                        logging.info('e2')
                        logging.info(e)
                        self.dead = True
                else:
                    self.fetch_nodes.append(self.evaluated[self.active_nodes[-self.genome_output_count-1]])
                    self.genome_output_values, val_train = self.tensorflow_preprocess(self.fetch_nodes, self.feed_dict, data_pair)
                    if self.apply_to_val:
                        self.validation_pair_output = (val_train, validation_pair[1])
        self.need_evaluate = False
        # clean up all tensorflow variables so that individual can be deepcopied
        # tensorflow values need not be deepcopy-ed because they're regenerated in evaluate anyway
        # this fixes the universe.py run_universe deepcopy() bug
        self.rec_clear()
        gc.collect()

   

    def non_tensorflow_evaluate(self, block_inputs, validation_pair):
        for i, input_ in enumerate(block_inputs):
            self.val_evaluated[-1*(i+1)] = validation_pair[0]
            self.evaluated[-1*(i+1)] = input_
        self.calculate_func_args_inputs()
        if not self.dead:
            for output_node in range(self.genome_main_count, self.genome_main_count+self.genome_output_count):
                referenced_node = self[output_node]
                if self.apply_to_val:
                    self.validation_pair_output = (self.val_evaluated[referenced_node], validation_pair[1]) #reall this should be append
                logging.info(self.labels)
                self.genome_output_values = (self.evaluated[referenced_node], self.labels) #Is there ever a time we have multiple multiple_genome_output_values
        self.need_evaluate = False #taking away append will break something
        self.rec_clear()
        gc.collect()


    def evaluate(self, block_inputs, labels_all, validation_pair):
        """
        :type block_inputs: object
        :type labels_all: object
        :type validation_pair: object
        """
        self.resetEvalAttr()
        self.findActive()
        self.labels = labels_all
        logging.debug("evaluate shape")
        logging.debug(self.labels.shape)
        logging.info("Active nodes {}".format(self.active_nodes))
        arg_values = np.array(self.args)
        for active_node in self.active_nodes:
            fn = self[active_node]
            if active_node < 0:
                # nothing to evaluate at input nodes
                logging.info('function at: {} is: {}'\
                    .format(active_node, fn))
                continue
            elif active_node >= self.genome_main_count:
                # nothing to evaluate at output nodes
                logging.info('function at: {} is: {} -> likely an output node'\
                    .format(active_node, fn))
                continue
            logging.info('function at: {} is: {} and has arguments: {}'\
                    .format(active_node, fn, arg_values[fn['args']]))

        logging.info('block_input: {}'.format(np.array(block_inputs).shape))
        if self.tensorblock_flag:
            self.tensorflow_evaluate(block_inputs, labels_all, validation_pair)
        else:
            self.non_tensorflow_evaluate(block_inputs, validation_pair)
        self.rec_clear()
        gc.collect()
    def rec_clear(self):
        self.graph = None
        self.feed_dict = {}
        self.fetch_nodes = []
        self.evaluated = [None] * self.genome_count
        self.dataset.clear_batch()
    #    self.genome_output_values = []
    #    self.validation_pair_output = []
        self.labels = []
        self.evaluated = [None] * self.genome_count
        self.val_evaluated = [None] * self.genome_count
        self.dataset.clear_batch() #for batch updates    
        tf.keras.backend.clear_session()