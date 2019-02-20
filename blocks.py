
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
#from fintess import Fitness
import operators
import arguments

import tensorflow as tf


class Block(Mate, Mutate):
    """
    Define the 'subclass' for a genome.

    Input nodes -> Block Instance -> Block Instance -> ... -> Output Nodes
    """

    def __init__(self, nickname,
                 setup_dict_ftn, setup_dict_arg, setup_dict_mate, setup_dict_mut,
                 operator_dict, block_input_dtypes, block_outputs_dtypes, block_main_count, block_arg_count,
                 block_mut_prob, block_mate_prob,
                 tensorblock_flag=False, learning_required=False, num_classes=None, batch_size=None):
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
        self.logs_path = tempfile.mkdtemp()
        self.learning_required = learning_required #
        self.num_classes = 10 # number of classes for what we are trying to classify...only relevant if there is supposed to be a learner
        self.need_evaluate = True # all new blocks need to be evaluated
        self.batch_size = batch_size # takes the batch_size from the block skeleton

        # Block - Argument List
        self.arg_methods = list(setup_dict_arg.keys())
        self.arg_weights = self.buildWeights('arg_methods', setup_dict_arg)
        self.fillArgs()

        # Block - Genome List - Input Nodes
        #  should already be set by Genome()
        # Block - Genome List - Main Nodes
        self.ftn_methods = list(setup_dict_ftn.keys())
        self.ftn_weights = self.buildWeights('ftn_methods', setup_dict_ftn)
        self.operator_dict = operator_dict
        # for key, val in operator_dict.items():
        #     print(key, val)
        # print(operator_dict)
        # quit()
        self.fillGenome(operator_dict)

        # Block - Evaluation
        self.resetEvalAttr()

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
        self.has_learner = False
        self.evaluated = [None] * self.genome_count
        self.genome_output_values = []
        if self.tensorblock_flag:
            self.graph = tf.Graph()
            self.feed_dict = {}
            self.fetch_nodes = []
#            with self.graph.as_default():
 #               saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

        else:
            pass

    def initialize_batch(self, x_train, y_train):
        self.__index_in_epoch = 0
        self._index_in_epoch = 0
        self._num_examples = len(x_train)
        self._epochs_completed = 0
        self._images = x_train
        self._labels = y_train

    "https://github.com/tensorflow/tensorflow/blob/7c36309c37b04843030664cdc64aca2bb7d6ecaa/tensorflow/contrib/learn/python/learn/datasets/mnist.py#L160"
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            #np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

    def tensorblock_evaluate(self, fetch_nodes, feed_dict, data_pair):
        # initialize variables needed for batch feeding
        self.initialize_batch(data_pair['x_train'], data_pair['y_train'])
        # final_outputs = []
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            # get placeholder nodes that will be fed to
            x_batch = tf.get_default_graph().get_operation_by_name('x_batch').outputs[0]
            y_batch = tf.get_default_graph().get_operation_by_name('y_batch').outputs[0]

            # X_train, y_train = data_pair['x_train'], data_pair['y_train']
            # print("shapes:", X_train.shape, y_train.shape)

            n_epochs = 2 # number of epochs to run for while training
            batch_size = self.batch_size # size of the batch
            return_outputs = []
            for epoch in range(n_epochs):
                epoch_loss = 0 # holds cumulative loss over the epoch
                # will hold predictions for training data at this epoch
                epoch_outputs = []
                print("num examples", self._num_examples)
                for step in range(int(np.ceil(self._num_examples/batch_size))):
                    X_train, y_train = self.next_batch(min(self._num_examples \
                        - step * batch_size, batch_size))
                    # print("shapes:", X_train.shape, y_train.shape)
                    feed_dict[x_batch] = X_train
                    feed_dict[y_batch] = y_train
                    tf_outputs = sess.run(
                                    fetches=fetch_nodes,
                                    feed_dict=feed_dict)

                    tf_output_dict = tf_outputs[0]
                    step_loss = tf_output_dict['loss']
                    print("epoch: {} loaded batch index: {}. Fed {}/{} samples. Step loss: {}"\
                        .format(epoch, step, step * batch_size, self._num_examples, step_loss))
                    # print('step_loss: ', step_loss)
                    # epoch_loss += step_loss
                    # print('at step: {} received tf_outputs with keys: {} and loss: {}'\
                    #     .format(step, tf_output_dict.keys(), tf_output_dict['loss']))
                    # print the class predictions for this run
                    # print('predictions have type: {} shape: {} and are: {}'\
                    #     .format(type(tf_output_dict['classes']),\
                    #     tf_output_dict['classes'].shape, tf_output_dict['classes']))
                    final_outputs = tf_output_dict['classes']
                    epoch_outputs += final_outputs.tolist()

                return_outputs = epoch_outputs # holds the outputs of the latest epochs
                print('Epoch completed. epoch_loss: ', epoch_loss)

                print(return_outputs)
            return np.array(return_outputs)

    def tensorboard_show(self, wait_seconds=60):
        # only should work after tensorblock_evaluate() is run
        cmd = "tensorboard --log_dir=%s" % self.logs_path
        args = shelx.split(cmd)
        p = subprocess.Popen(args)
        print("tensorboard link created for %iseconds" % wait_seconds)
        time.sleep(wait_seconds)
        p.terminate() # or p.kill()
        print("tensorbard killed")


    def evaluate(self, block_inputs, labels_all):
        self.resetEvalAttr()
        self.findActive()
        if (self.learning_required) and (not self.has_learner):
            # didn't learn, zero fitness
            print("didn't learn, zero fitness")
            self.dead = True
        else:
            print('block_input: {}'.format(np.array(block_inputs).shape))
            data_pair = {}
            for i, input_ in enumerate(block_inputs): #self.genome_input_dtypes):
                if self.tensorblock_flag:
#                    self.evaluated[-1*(i+1)] = input_
                    print("self.evaluated: ", self.evaluated)

                    # consider reading in the dataset with slices..."from_tensor_slices"
                    # then dataset.shuffle.repate.batch and dataset.make_one_shot_iterator
                    data_dimension = list(input_.shape)
                    data_dimension[0] = None # variable input size, "how to tell tensorflow" to figure it out by def.

                    # print(data_dimension)
                    with self.graph.as_default():
                        # TODO, verify that this placeholder works
                        # self.evaluated[-1*(i+1)] = tf.placeholder(tf.float32, data_dimension)
                        batch_X = tf.placeholder(tf.float32, data_dimension, name='x_batch')
                        self.evaluated[-1*(i+1)] = batch_X

                    self.feed_dict[batch_X] = input_
                    data_pair['x_train'] = input_
                    data_pair['y_train'] = labels_all
                    # self.feed_dict = {batch_X: input_}

                else:
                    self.evaluated[-1*(i+1)] = input_
            for node_index in self.active_nodes:
                if node_index < 0:
                    # nothing to evaluate at input nodes
                    continue
                elif node_index >= self.genome_main_count:
                    # nothing to evaluate at output nodes
                    continue
                else:
                    # only thing left is main node...extract "ftn", "inputs", and "args" below
                    pass
                # get function to evaluate
                function = self[node_index]["ftn"]
                # get inputs to function to evaluate
                inputs = []
                node_input_indices = self[node_index]["inputs"]
                for node_input_index in node_input_indices:
                    inputs.append(self.evaluated[node_input_index])
                # get arguments to function to evaluate
                args = []
                node_arg_indices = self[node_index]["args"]
                for node_arg_index in node_arg_indices:
                    args.append(self.args[node_arg_index])
                # and evaluate
                try:
                    if self.tensorblock_flag:
                        with self.graph.as_default():
                            # really we are building the graph here; we need to evaluate after it is fully built
                            self.evaluated[node_index] = function(*inputs, *args)
                    else:
                        self.evaluated[node_index] = function(*inputs, *args)
                except Exception as e:
                    raise(e)
                    self.dead = True
                    break
            if not self.dead:
                if self.tensorblock_flag:
                    # final touches in the graph
                    with self.graph.as_default():
                        for output_node in range(self.genome_main_count, self.genome_main_count+self.genome_output_count):
                            # logits = tf.layers.dense(inputs=self.evaluated[self[output_node]], units=self.num_classes) # logits layer
                            # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            #                             logits,
                            #                             tf.cast(labels,dtype=tf.float32)))
                            print("self[output_node]:", self[output_node])
                            print("self.evaluated[self[output_node]]:", self.evaluated[self[output_node]])
                            # flatten input matrix to meet NN output size (numinstances, numclasses)
                            flattened = tf.layers.Flatten()(self.evaluated[self[output_node]])
                            print(flattened)
                            labels = tf.placeholder(tf.int32, [None], name='y_batch')
                            logits = tf.layers.dense(inputs=flattened, units=self.num_classes) # logits layer
                            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                                        logits = logits,
                                                        labels = tf.one_hot(indices=labels, depth=self.num_classes, dtype=tf.float32)))
                            # predictions = tf.nn.softmax(logits=logits)
                            # or tf.losses.sparse_softmax_cross_entropy ...didn't put a lot of thought in this really
                            tf.summary.tensor_summary("loss", loss)
                            # self.evaluated[output_node] = {
                            #     "classes": tf.argmax(input=logits, axis=1),
                            #     "probabilities": tf.nn.softmax(logits)}
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
                        # print(loss)
                    for graph_metadata in [train_step, merged_summary]: # opportunity to add other things we would want to fetch from the graph
                        # remember, we need 'train_step' so that the optimizer is run; we don't actually need the output
                        self.fetch_nodes.append(graph_metadata)
                    try:
                        # now that the graph is built, we evaluate here
                        # print("self.fetch_nodes")
                        # print(self.fetch_nodes)
                        # print("self.feed_dict")
                        # print(self.feed_dict)
                        self.genome_output_values = self.tensorblock_evaluate(self.fetch_nodes, self.feed_dict, data_pair)
                    except Exception as e:
                        raise(e)
                        #print(e)
                        self.dead = True
                else:
                    # if it's not tensorflow
                    for output_node in range(self.genome_main_count, self.genome_main_count+self.genome_output_count):
                        referenced_node = self[output_node]
                        self.genome_output_values.append(self.evaluated[referenced_node])
            else:
                pass
            #self.evaluated = None # clear up some space by deleting eval from memory
            self.need_evaluate = False
            gc.collect()
