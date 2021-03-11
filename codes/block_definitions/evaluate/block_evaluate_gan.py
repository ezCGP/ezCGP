from abc import ABC, abstractmethod
from copy import deepcopy

from torch import nn

from codes.block_definitions.evaluate.block_evaluate import BlockEvaluate_Abstract
from codes.block_definitions.utilities.operators_pytorch import InputLayer
from codes.genetic_material import BlockMaterial
from codes.utilities.custom_logging import ezLogging
from data.data_tools.simganData import SimGANDataset

class BlockEvaluate_PyTorch_Abstract(BlockEvaluate_Abstract):
    """
    An abstract class defining the structure of a PyTorch neural network BlockEvaluate. Provides a function to build a PyTorch nueral network graph
    from PyTorchLayerWrapper operators.
    """
    @abstractmethod
    def __init__(self):
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

        # Add a layer to get the output back into the right shape
        # TODO: consider using a linear layer instead of a conv1d and then unflattening. Not sure what is the right move
        extra_layers = []
        if len(output_shape) == 1:
            extra_layers.append(nn.Unflatten(dim=0, unflattened_size=(1,)))
        # Tanh to keep our output in the range of 0 to 1
        extra_layers = extra_layers + [nn.Conv1d(output_shape[0], 1, kernel_size=1), nn.Tanh()]

        block_material.graph = nn.Sequential(graph, *extra_layers)

        return block_material.graph
       
    def evaluate(self,
                 block_material,
                 block_def,
                 training_datalist,
                 validating_datalist,
                 supplements):
        '''
        All we have to do in evaluate is build the graph.
        '''
        ezLogging.info("%s - Start evaluating..." % (block_material.id))

        try:
            supplements.append(self.build_graph(block_material, block_def, training_datalist[0]))
        except Exception as err:
            ezLogging.critical("%s - Build Graph; Failed: %s" % (block_material.id, err))
            block_material.dead = True
            import pdb; pdb.set_trace()
            return

        block_material.output = [training_datalist, validating_datalist, supplements]


class BlockEvaluate_SimGAN_Discriminator(BlockEvaluate_PyTorch_Abstract):
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_SimGAN_Discriminator Class" % (None, None))

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

        return block_material.graph

    def evaluate(self,
                 block_material,
                 block_def,
                 training_datalist,
                 validating_datalist,
                 supplements=None):
        '''
        All we have to do in evaluate is build the graph.
        '''
        ezLogging.info("%s - Start evaluating..." % (block_material.id))
        try:
            supplements.append(self.build_graph(block_material, block_def, training_datalist[0]))
        except Exception as err:
            ezLogging.critical("%s - Build Graph; Failed: %s" % (block_material.id, err))
            block_material.dead = True
            import pdb; pdb.set_trace()
            return

        block_material.output = [training_datalist, validating_datalist, supplements]

class BlockEvaluate_SimGAN_Train_Config(BlockEvaluate_Abstract):
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_SimGAN_Train_Config Class" % (None, None))

    def evaluate(self,
                 block_material,
                 block_def,
                 training_datalist,
                 validating_datalist,
                 supplements=None):
        '''
        Simply return the config
        '''
        ezLogging.info("%s - Start evaluating..." % (block_material.id))
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

        supplements.append(output_list[0]) # this adds the config to supplements

        block_material.output = [training_datalist, validating_datalist, supplements]