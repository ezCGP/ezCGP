'''
PyTorch has a very unique way of building/defining a graph/network.
So we decided to have a separate block_evaluate script to store all PyTorch related blocks.
Checkout misc/play_with_torch_network.py to see our work in playing with how we can abstract
out how to build a graph/network given some genome.
'''

### packages
from abc import ABC, abstractmethod
from copy import deepcopy
from torch import nn
from collections import OrderedDict

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.block_definitions.evaluate.block_evaluate import BlockEvaluate_Abstract
from codes.genetic_material import BlockMaterial
from codes.utilities.custom_logging import ezLogging
#from data.data_tools.simganData import SimGANDataset



class MyTorchNetwork(nn.Module):
    '''
    have to use nn.Module.add_module() method to make sure the parameters() method
    returns the modules we pulled from the genome.

    STRONG assumption that all primitives will return an instance of some class that
    has a __call__() and get_out_shape() methods...kinda sucks to make those primitives
    but making a pytorch graph that isn't sequential is really difficult, mostly
    because you have to manually specify the input layer shape when defining a layer.

    ASSUMPTION: that our network will only ever have 1 input tensor x,
    so block_def.input_count will always be 1.

    If we want to couple layers together (like conv layer with activation) then use
    the same primitive format but return an instance of nn.Sequential.

    TODO: need to pass data/input tensor shape for first few primitives to use!
    '''
    def __init__(self, block_material, block_def, input_shape, final_module_dicts):
        super().__init__()
        self.graph_connections = OrderedDict()
        self.graph_connections[-1] = {"module": None,
                                      "inputs": [],
                                      "output_shape": input_shape}

        for node_index in block_material.active_nodes:
            if node_index < 0:
                # do nothing. at input node
                continue
            elif node_index >= block_def.main_count:
                # do nothing.
                continue
            else:
                # main node. this is where we evaluate
                input_connections = block_material[node_index]['inputs']
                input_shapes = []
                for input_index in input_connections:
                    input_dict = self.graph_connections[input_index]
                    input_shapes.append(input_dict['output_shape'])

                module_name = 'node_%i' % node_index
                module = block_material[node_index]['function']
                args = block_material[node_index]['args']
                ezLogging.debug("%s - Builing %s; Function: %s, Inputs: %s, Shapes: %s, Args: %s" % (block_material.id,
                                                                                                     module_name,
                                                                                                     module,
                                                                                                     input_connections,
                                                                                                     input_shapes,
                                                                                                     args))
                print(node_index, module)
                module_instance = module(input_shapes, *args)
                if isinstance(module_instance, nn.Module):
                    self.add_module(name=module_name,
                                    module=module_instance)
                    self.graph_connections[node_index] = {"module": module_name,
                                                          "inputs": input_connections,
                                                          "output_shape": module_instance.get_out_shape()}
                else:
                    self.graph_connections[node_index] = {"module": module_instance,
                                                          "inputs": input_connections,
                                                          "output_shape": module_instance.get_out_shape()}

        # add any final modules...going to assume it is always nn.Module type
        for i, module_dict in enumerate(final_module_dicts):
            module = module_dict["module"]
            args = module_dict["args"]
            if i == 0:
                input_shapes = [self.graph_connections[node_index]["output_shape"]]
            else:
                input_shapes = [module_instance.get_out_shape()]
            module_instance = module(input_shapes, *args)
            self.add_module(name="final_module_%i" % i,
                            module=module_instance)

        self.final_layer_shape = module_instance.get_out_shape()
        self.genome_length = block_def.genome_count


    def forward(self, x):
        evaluated_connections = [None]*self.genome_length

        # pass in input...for now we as
        evaluated_connections[-1] = x

        for node_index, node_dict in self.graph_connections.items():
            if node_index < 0:
                 continue

            if isinstance(node_dict["module"], str):
                # then it is a nn.Module so grab directly from self
                callable_module = self._modules[node_dict["module"]]
            else:
                callable_module = node_dict["module"]

            input_connections = node_dict["inputs"]
            inputs = []
            for input_node in input_connections:
                inputs.append(evaluated_connections[input_node])

            evaluated_connections[node_index] = callable_module(*inputs)

        # evaluate any final layers
        if "final_module_0" in self._modules:
            callable_module = self._modules["final_module_0"]
            inputs = [evaluated_connections[node_index]]
            output = callable_module(*inputs)
            final_module_count = 1
            while True:
                module_name = "final_module_%i" % final_module_count
                if module_name in self._modules:
                    callable_module = self._modules[module_name]
                    output = callable_module(output)
                    final_module_count+=1
                else:
                    break
        else:
            output = evaluated_connections[node_index]

        return output



class BlockEvaluate_PyTorch_Abstract(BlockEvaluate_Abstract):
    '''
    Outline a parent class to build out PyTorch Networks.
    '''
    def __init__(self):
        # TODO, verify if these are even needed
        #globals()['torch'] = importlib.import_module('torch')
        #globals()['nn'] = importlib.import_module('torch.nn')
        self.final_module_dicts = []

    @abstractmethod
    def build_graph(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


    def standard_build_graph(self,
                             block_material: BlockMaterial,
                             block_def,#: BlockDefinition,
                             input_layers):
        # always assuming one input so len(input_layers) should be 1
        input_shape = input_layers[0].shape
        self.graph = MyTorchNetwork(block_material, block_def, input_shape, self.final_module_dicts)
        ezLogging.info("%s - Ending build graph" % (block_material.id))



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
        extra_layers = [nn.Flatten(start_dim=1), nn.Linear(in_features, 2)]

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