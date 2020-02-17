from blocks import Block
import Augmentor
import traceback
from mutate_methods import Mutate
import numpy as np
import gc
class AugmentorBlock(Block):
    """
    Define the 'subclass' for a genome.
    Input nodes -> Block Instance -> Block Instance -> ... -> Output Nodes
    """

    def __init__(self, nickname,
                setup_dict_ftn, setup_dict_arg, setup_dict_mut,
                operator_dict, block_input_dtypes, block_outputs_dtypes,
                block_main_count, block_arg_count,
                block_mut_prob,
                large_dataset=None):

        Mutate.__init__(self, block_input_dtypes, block_outputs_dtypes,
                              block_main_count, block_arg_count)
        self.mut_prob = block_mut_prob
        self.mut_methods = list(setup_dict_mut.keys())
        self.mut_weights = self.buildWeights('mut_methods', setup_dict_mut)
        self.mut_dict = setup_dict_mut

        #Fitness.__init__(self)

        # Block - MetaData
        self.name = nickname
        self.need_evaluate = True # all new blocks need to be evaluated
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
        self.evaluated = [None] * self.genome_count
        self.dead = False
        pass
        # Block - Evaluation
        #self.resetEvalAttr()

        #self.dataset = DataSet([], []) #this is for feeding the data in batches

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
            node_input_indices = self[node_index]["inputs"]
            for node_input_index in node_input_indices:
                inputs.append(self.evaluated[node_input_index])
            args = []
            node_arg_indices = self[node_index]["args"]
            for node_arg_index in node_arg_indices:
                args.append(self.args[node_arg_index].value)
            try:
                self.evaluated[node_index] = function(*inputs, *args)
            except Exception as e:
                self.dead = True
                print('calculate_func_args_inputs error: ')
                print(traceback.format_exc())
                print(e)
                break


    def evaluate(self):
        p = Augmentor.Pipeline()
        #for i, input_ in enumerate(block_inputs):
        self.evaluated[-1] = p
        self.findActive()
        self.calculate_func_args_inputs()

        if not self.dead:
            for output_node in range(self.genome_main_count, self.genome_main_count+self.genome_output_count):
                referenced_node = self[output_node]
                self.genome_output_values = self.evaluated[referenced_node]

        self.need_evaluate = False #taking away append will break something
     #   self.rec_clear()
        gc.collect()
        print("hello")
        """
        :type block_inputs: object
        :type labels_all: object
        :type validation_pair: object
        """
        #self.resetEvalAttr()
        self.findActive()
       # self.labels = labels_all
        # logging.debug("evaluate shape")
        # logging.debug(self.labels.shape)
        print("Active nodes {}".format(self.active_nodes))
        arg_values = np.array(self.args)
        for active_node in self.active_nodes:
            fn = self[active_node]
            if active_node < 0:
                # nothing to evaluate at input nodes
                print('function at: {} is: {}' \
                      .format(active_node, fn))
                continue
            elif active_node >= self.genome_main_count:
                # nothing to evaluate at output nodes
                print('function at: {} is: {} -> likely an output node' \
                      .format(active_node, fn))
                continue
            print('function at: {} is: {} and has arguments: {}' \
                  .format(active_node, fn, arg_values[fn['args']]))

        print('block_input: {}'.format(p))
        # if self.tensorblock_flag:
        #     self.tensorflow_evaluate(block_inputs, labels_all, validation_pair)
        # else:
        #     self.non_tensorflow_evaluate(block_inputs, validation_pair)
       # self.rec_clear()
        gc.collect()