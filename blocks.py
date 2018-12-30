
# external packages
import numpy as np
import time
import shlex
import subprocess
import os

# my scripts
from mate_methods import Mate
from mutate_methods import Mutate
from fintess import Fitness
import operators
import arguments


class Block(Mate, Mutate, Fitness):
    """
    Define the 'subclass' for a genome.

    Input nodes -> Block Instance -> Block Instance -> ... -> Output Nodes
    """

    def __init__(self, nickname,
                 ftn_dict, arg_dict, mate_dict, mut_dict,
                 gene_dict, block_inputs, block_outputs, block_main_count, block_arg_count,
                 tensor_block=False):
        """
        take in a dictionary for each: key is the method, and value gives how to often that method will show
        values should go from (0,1]; anything geq to 1 means that we want this equally distributed with whatever is left.
        the sum of values less than 1 should not be greater than 1 or it will error
        """

        # Block - MetaData
        self.name = nickname
        self.tensor_block = tensor_block
        self.logs_path = tempfile.mkdtemp()
        self.num_classes = 10 # number of classes for what we are trying to classify
        self.need_evaluate = True # all new blocks need to be evaluated

        # Block - Argument List
        self.args_count = block_arg_count
        self.args = [None]*block_arg_count
        self.arg_methods = list(arg_dict.keys())
        self.arg_weights = self.buildWeights('arg_methods', arg_dict)
        start_point = 0
        end_point = 0
        for arg_index, arg_dtype in enumerate(self.arg_methods):
            end_point += round(self.arg_weights[arg_index]*block_arg_count)
            self.args[start_point:end_point] = arg_dtype # need some way to initialize the args here...call mutate?
            start_point = end_point
        if end_point != block_arg_count:
            # some wierd rounding error then...just go ahead and asign the last few with the previous batch
            self.args[end_point:block_arg_count] = arg_dtype # ...use method from above
        else:
            pass

        # Block - Genome List
        self.block_genome = [None]*(len(block_inputs)+block_main_count+len(block_outputs))
        self.block_genome_count = len(self.genome)

        # Block - Genome List - Input Nodes
        self.block_inputs_dtypes = block_inputs # a list of data types. the exact values will be assigned at evaluation step
        self.block_inputs_count = len(block_inputs)
        self.block_genome[-1*len(block_inputs):] = ["InputPlaceholder"]*len(block_inputs)#block_inputs

        # Block - Genome List - Main Nodes
        self.block_main_count = block_main_count
        self.ftn_methods = list(ftn_dict.keys())
        self.ftn_weights = self.buildWeights('ftn_methods', ftn_dict)
        for node_index in range(block_main_count):
            # randomly select all function, find a previous node with matching output_dtype matching our function input_dtype
            # iterate through each function until you find matching output and input data types
            ftns = self.randomFtn()
            for ftn in ftns:
                # connect to a previous node
                input_dtypes = gene_dict[ftn]["inputs"]
                input_nodes = [] # [None]*len(input_dtypes)
                for input_dtype in input_dtypes:
                    input_nodes.append(self.randomInput(max_=node_index, dtype=input_dtype))
                if None in input_nodes:
                    # couldn't find a matching input + output value. try the next ftn
                    break
                else:
                    # go on to find args
                    pass
                # connect to any required arguments
                arg_dtypes = gene_dict[ftn]["args"]
                arg_nodes = []
                for arg_dtype in arg_dtypes:
                    arg_nodes.append(self.randomArg(dtype=arg_dtype))
                # assign all values to the genome
                self.block_genome[node_index] = {"ftn" : ftn,
                                                 "inputs" : input_nodes,
                                                 "args" : arg_nodes}
                # we found a ftn that works, move on to next node_index
                break

        # Block - Genome List - Outputs Nodes
        self.block_outputs_dtypes = block_outputs # a list of data types. the exact values will be evaluated at evaluation step
        self.block_outputs_count = len(block_outputs)
        for i, output_dtype in enumerate(block_outputs):
            self.block_genome[block_main_count+i] = self.randomInput(min_=0, max_=block_main_count, dtype=output_dtype)

        self.mate_methods = list(mate_dict.keys())
        self.mate_weights = self.buildWeights('mate_methods', mate_dict)
        self.mate_dict = mate_dict
        self.block_mate_prob = mate_prob

        self.mut_methods = list(mut_dict.keys())
        self.mut_weights = self.buildWeights('mut_methods', mut_dict)
        self.mut_dict = mut_dict
        self.block_mut_prob = mutate_prob
        
        # Block - Evaluation
        self.resetEvalAttr()


    def __setitem__(self, node_index, dict_):
        self.block_genome[node_index] = dict_


    def __getitem__(self, node_index):
        return self.block_genome[node_index]


    def buildWeights(self, method, method_dict):
        prob_remaining = 1.0
        weights = [0] * len(method_dict)
        equally_distribute = []
        for i, m in enumerate(self.__dict__[method]):
            if method_dict[m]['prob'] <= 0:
                # will never call this method
                # setting weight to 0 instead of removing
                # ...my guess is that it's more useful to see that it's set to 0 rather than gone
                pass
            elif method_dict[m]['prob'] < 1:
                # assign it that prob
                prob_remaining -= method_dict[m]['prob']
                if prob_remaining < 0:
                    # error, sum of prob is greater than 1
                    print("UserInputError: current sum of prob/weights for %s is > 1" % method)
                    exit()
                else:
                    weights[i] = method_dict[m]['prob']
            else:
                # user wants this prob to be equally distributed with whatever is left
                equally_distribute.append(i)
        # we looped through all methods, now equally distribute the remaining amount
        if len(equally_distribute) > 0:
            eq_weight = round(prob_remaining/len(equally_distribute), 4)
            for i in equally_distribute:
                weights[i] = eq_weight
        else:
            pass
        # now clean up any rounding errors by appending any remainder to the last method
        remainder = 1 - sum(weights)
        if remainder > .01:
            print("UserInputError: total sum of prob/weights for %s is < .99" % method)
            exit()
        else:
            weights[-1] += remainder
        return weights


    def randomFtn(self, only_one=False, exclude=None):
        choices = self.ftn_methods
        weights = self.ftn_weights
        if exclude is not None:
            for val in exclude:
                choices = np.delete(choices, np.where(choices==val))
                weights = np.delete(weights, np.where(choices==val))
        else:
            pass
        if only_one:
            return np.random.choice(a=choices, size=1, p=weights)
        else:
            return np.random.choice(a=choices, size=len(self.ftn_methods), replace=False, p=weights)


    def randomInput(self, dtype, min_=-1*self.block_inputs_count, max_=self.block_main_count+1, exclude=None):
        # max_ is one above the largest integer to be drawn
        # so if we are randomly finding nodes prior to a given node index, set max_ to that index
        choices = np.arange(min_, max_)
        if exclude is not None:
            for val in exclude:
                choices = np.delete(choices, np.where(choices==val))
        else:
            pass
        possible_nodes = np.random.choice(a=choices, size=max_-min_, replace=False)
        # iterate through each input until we find a datatype that matches dtype
        for poss_node in possible_nodes:
            poss_node_outputdtype = self.getNodeType(node_index=poss_node, output_dtype=True)
            # note, we currently only allow for arity of 1 for our ftns...so getNodeType will return a value not a list if output_dtype is True
            if dtype == poss_node_outputdtype:
                return poss_node
            else:
                pass
        # if we got this far then we didn't find a match
        return None


    def randomArg(self, dtype, exclude=None):
        choices = []
        for arg_index in range(self.args_count):
            if arg_index in exclude:
                continue
            else:
                pass
            arg_dtype = type(self.args[arg_index]).__name__
            if dtpye == arg_dtype:
                choices.append(arg_index)
        if len(choices) == 0:
            print("UserInputError: A ftn was provided without having its required data type in the arguments")
            exit()
        else:
            return np.random.choice(a=choices, size=1)


    def getNodeType(self, node_index, input_dtype=False, output_dtype=False):
        if node_index < 0:
            # then it's a Block Input Node
            return self.block_inputs_dtypes[-1*node_index-1] # -1-->0, -2-->1, -3-->2
        elif node_index >= self.block_main_count:
            # then it's a Block Output Node
            return self.block_outputs_dtypes[node_index-self.block_main_count]
        else:
            # then it's a Block Main Node
            pass 
        ftn = self.genome[node_index]
        if input_dtype:
            # get the required input data types for this function
            return gene_dict[ftn]["inputs"] # will return a list
        elif output_dtype:
            # get the output data types for this function
            return gene_dict[ftn]["outputs"] # returns a single value, not a list...arity is always 1 for outputs
        else:
            print("ProgrammerError: script writer didn't assign input/output dtype for getNodeType method")
            exit()


    def mutate(self):
        roll = np.random.random()
        if roll <= self.block_mut_prob:
            # then mutate
            mutate_method = np.random.choice(a=self.mut_methods, size=1, p=self.mut_weights)
            mutate_method(self, *self.mut_dict[mutate_method]['args']) # or try Mutate.mut_method
            # make sure in mutate_methods.py we have class Mutate() where we define all these methods
            self.need_evaluate = True
        else:
            # don't mutate
            pass


    def mate(self, other):
        roll = np.random.random()
        if roll <= self.block_mate_prob:
            # then mate
            mate_method = np.random.choice(a=self.mate_methods, size=1, p=self.mate_weights)
            offspring_list = mate_method(self, other, *self.mate_dict[mate_method]['args'])
            for offspring in offspring_list:
                offspring.need_evaluate = True
            return offspring_list
        else:
            # don't mate
            return None


    def resetEvalAttr(self):
        self.dead = False
        self.has_learner = False
        self.evaluated = [None] * self.block_genome_count
        self.block_outputs_values = []
        if self.tensor_block:
            self.graph = tf.Graph()
            self.feed_dict = {}
            self.fetch_nodes = []
            with self.graph.as_default():
                saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
        else:
            pass


    def findActive(self):
        """
        Acitve nodes should include:
        * all OUTPUT nodes
        * any INPUT node eventually connected to output (expressed as negative positions: -1, -2, etc)
        * any MAIN node eventually connected to output
        """
        self.active_nodes = set(range(self.block_main_count, self.block_main_count+self.block_outputs_count))
        self.active_ftns = set()
        self.active_args = set()
        # update with which nodes feed into the block outputs
        for node_input in self.active_nodes:
            self.active_nodes.update(self[node_index])
        # update what feed into the block main nodes
        for node_index in reversed(range(self.block_main_count)):
            if node_index in self.active_nodes:
                # check which nodes are input to this node_index...then update our active_nodes set list
                self.active_nodes.update(self[node_index]["inputs"])
                self.active_ftns.update([self[node_index]["ftn"]])
                if len(self[node_index]["args"]) > 0:
                    self.active_args.update(self[node_index]["args"])
                else: # no args to update
                    pass
                if (not self.has_learner) and (self[node_index]["ftn"].__name__=="single_learner"):
                    self.has_learner = True
                else: # already learned or ftn isn't a learner
                    pass
            else: # not an active node; don't care
                pass
        self.active_nodes = sorted(list(self.active_nodes))
        self.active_ftns = sorted(list(self.active_ftns))
        self.active_args = sorted(list(self.active_args))


    def tensorblock_evaluate(self, fetch_nodes, feed_dict):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            self.tlog_writer = tf.summary.FileWriter(self.logs_path, graph=sess.graph) # tensorboard uses these logs
            self.tlog_writer.add_graphs(sess.graph)
            for step in STEPS: # TODO how and where to define number of steps to train?
                tf_outputs = sess.run(
                                fetches=fetch_nodes,
                                feed_dict=feed_dict)
                self.tlog_writer.add_summary(summary, step)
                saver.save(sess, save_path=self.logs_path, global_step=step) # keep the 5 most recent steps and keep one from ever hour
            self.tlog_writer.close()


    def tensorboard_show(self, wait_seconds=60):
        # only should work after tensorblock_evaluate() is run
        cmd = "tensorboard --log_dir=%s" % self.logs_path
        args = shelx.split(cmd)
        p = subprocess.Popen(args)
        print("tensorboard link created for %iseconds" % wait_seconds)
        time.sleep(wait_seconds)
        p.terminate() # or p.kill()
        print("tensorbard killed")


    def evaluate(self, block_inputs, learning_required=True):
        self.resetEvalAttr()
        self.findActive()
        if (learning_required) and (not self.has_learner):
            # didn't learn, zero fitness
            self.dead = True
        else:
            for i, input_ in enumerate(block_inputs):
                if self.tensorblock:
                    self.feed_dict[self.evaluated[-1*(i+1)]] = input_
                    # consider reading in the dataset with slices..."from_tensor_slices"
                    # then dataset.shuffle.repate.batch and dataset.make_one_shot_iterator
                    data_dimension = input_.shape
                    with self.graph.as_default():
                        # TODO, verify that this placeholder works
                        self.evaluated[-1*(i+1)] = tf.placeholder(tf.float32, [None,data_dimension[1]])
                else:
                    self.evaluated[-1*(i+1)] = input_
            for node_index in self.active_nodes:
                if node_index < 0:
                    # nothing to evaluate at input nodes
                    continue
                elif node_index >= self.block_main_count:
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
                    if self.tensorblock:
                        with self.graph.as_default():
                            # really we are building the graph here; we need to evaluate after it is fully built
                            self.evaluated[node_index] = function(*inputs, *args)
                    else:
                        self.evaluated[node_index] = function(*inputs, *args)
                except:
                    self.dead = True
                    break
            if not self.dead:
                if self.tensorblock:
                    # final touches in the graph
                    with self.graph.as_default():
                        for ouptput_node in range(self.block_main_count: self.block_main_count+self.block_outputs_count):
                            logits = tf.layers.dense(inputs=self.evaluated[self[output_node]], units=self.num_classes) # logits layer
                            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                                        logits,
                                                        tf.cast(labels,dtype=tf.float32)))
                            # or tf.losses.sparse_softmax_cross_entropy ...didn't put a lot of thought in this really
                            tf.summary.tensor_summary("loss", loss)
                            self.evaluated[output_node] = {
                                "classes": tf.argmax(input=logits, axis=1),
                                "probabilities": tf.nn.softmax(logits)}
                            self.fetch_nodes.append(self.evaluated[output_node])
                            tf.summary.tensor_summary("classes", self.evaluated[output_node]["classes"])
                            tf.summary.tensor_summary("probabilities", self.evaluated[output_node]["probabilities"])
                        optimizer = tf.train.AdamOptimizer() # TODO add optimizer into 'arguments' to and apply GA to it for mutate + mate
                        #global_step = tf.Variable(0, name='backprop_steps', trainable=False)
                        train_step = optimizer.minimize(loss, global_step=step) #global_step)
                        #tf.summary.scalar('loss', loss)
                        #tf.summary.scalar('logits', logits)
                        #tf.summary.scalar('results', results)
                        merged_summary = tf.summary.merge_all()
                    for graph_metadata in [train_step, merged_summary]: # opportunity to add other things we would want to fetch from the graph
                        # remember, we need 'train_step' so that the optimizer is run; we don't actually need the output
                        self.fetch_nodes.append(graph_metadata)
                    try:
                        # now that the graph is built, we evaluate here
                        self.block_outputs_values = self.tensorblock_evaluate(self.fetch_nodes, self.feed_dict)
                    except:
                        self.dead = True
                else:
                    for ouptput_node in range(self.block_main_count: self.block_main_count+self.block_outputs_count):
                        referenced_node = self[output_node]
                        self.block_outputs_values.append(self.evaluated[referenced_node])
            else:
                pass
            self.evaluated = None # clear up some space by deleting eval from memory
            gc.collect()
        self.getFitness()

