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
import numpy as np
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
import codes.block_definitions.utilities.operators_pytorch as opPytorch



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
            try:
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
                    module = block_material[node_index]['ftn']

                    args = []
                    node_arg_indices = block_material[node_index]["args"]
                    for node_arg_index in node_arg_indices:
                        args.append(block_material.args[node_arg_index].value)

                    ezLogging.debug("%s - Builing %s; Function: %s, Inputs: %s, Shapes: %s, Args: %s" % (block_material.id,
                                                                                                         module_name,
                                                                                                         module,
                                                                                                         input_connections,
                                                                                                         input_shapes,
                                                                                                         args))
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
            except Exception as err:
                ezLogging.critical("crap #0 %s" % err)
                #import pdb; pdb.set_trace()
                raise Exception(err)

        # add any final modules...going to assume it is always nn.Module type
        self.final_linear_connections = [] # synonomous with graph_connections ordered dict but linear so just a list
        for i, module_dict in enumerate(final_module_dicts):
            try:
                module = module_dict["module"]
                module_name = "final_module_%i" % i
                args = module_dict["args"]
                if i == 0:
                    last_main_active = block_material[block_def.main_count]
                    input_shapes = [self.graph_connections[last_main_active]["output_shape"]]
                else:
                    input_shapes = [self.final_linear_connections[-1]["output_shape"]]

                ezLogging.debug("%s - Builing %s; Function: %s, Inputs: PrevFinal, Shapes: %s, Args: %s" % (block_material.id,
                                                                                                         module_name,
                                                                                                         module,
                                                                                                         input_shapes,
                                                                                                         args))
                module_instance = module(input_shapes, *args)
                if isinstance(module_instance, nn.Module):
                    self.add_module(name=module_name,
                                    module=module_instance)
                    final_module_dict = {"module": module_name,
                                         "output_shape": module_instance.get_out_shape()}
                else:
                    final_module_dict = {"module": module_instance,
                                         "output_shape": module_instance.get_out_shape()}
                self.final_linear_connections.append(final_module_dict)

            except Exception as err:
                ezLogging.critical("crap #1 %s" % err)
                #import pdb; pdb.set_trace()
                raise Exception(err)

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

            ezLogging.debug("%s - Network running 'forward' on %s" % ("id NA", callable_module))
            evaluated_connections[node_index] = callable_module(*inputs)
        
        # get output from most recent run
        outputs = evaluated_connections[node_index]

        # evaluate any final layers
        for node_index, node_dict in enumerate(self.final_linear_connections):
            if isinstance(node_dict["module"], str):
                # then it is a nn.Module so grab directly from self
                callable_module = self._modules[node_dict["module"]]
            else:
                callable_module = node_dict["module"]

            ezLogging.debug("%s - Network running 'forward' on %s" % ("id NA", callable_module))
            outputs = callable_module(outputs)

        return outputs



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
        block_material.graph = MyTorchNetwork(block_material, block_def, input_shape, self.final_module_dicts)
        ezLogging.info("%s - Ending build graph" % (block_material.id))


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



class BlockEvaluate_SimGAN_Refiner(BlockEvaluate_PyTorch_Abstract):
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_SimGAN_Refiner Class" % (None, None))
        super().__init__()

        in_channels = 1 # TODO hard coded...ish...i mean this is specific for simgan so it is okay...not like batch size
        self.final_module_dicts.append({"module": opPytorch.conv1d_layer,
                                        "args": [in_channels, 1, 1, 0, nn.Tanh()]})


    def build_graph(self, block_material, block_def, data):
        '''
        Build the SimGAN refiner network
        '''
        ezLogging.debug("%s - Building Graph" % (block_material.id))
        self.standard_build_graph(block_material, block_def, data)

        # Verify new images match shapes of input images
        if np.any(np.array(block_material.graph.final_layer_shape) != np.array(data[0].shape)):
            raise Exception("Output of Refiner %s doesn't match the image shape we exepct %s" % (block_material.graph.final_layer_shape, data[0].shape))


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
            input_images = training_datalist[0]
            self.build_graph(block_material, block_def, [input_images])
            #supplements.append(block_material.graph)
        except Exception as err:
            ezLogging.critical("%s - Build Graph; Failed: %s" % (block_material.id, err))
            block_material.dead = True
            #import pdb; pdb.set_trace()
            return

        block_material.output = [block_material.graph]



class BlockEvaluate_SimGAN_Discriminator(BlockEvaluate_PyTorch_Abstract):
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockEvaluate_SimGAN_Discriminator Class" % (None, None))
        super().__init__()

        self.final_module_dicts.append({"module": opPytorch.linear_layer,
                                        "args": [1]})
        self.final_module_dicts.append({"module": opPytorch.pytorch_squeeze,
                                        "args": []})
        self.final_module_dicts.append({"module": opPytorch.sigmoid_layer,
                                        "args": []})


    def build_graph(self, block_material, block_def, data):
        '''
        Build the SimGAN discriminator network
        '''
        ezLogging.debug("%s - Building Graph" % (block_material.id))
        self.standard_build_graph(block_material, block_def, data)


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
            # local discriminator
            if block_material.train_local_loss:
                local_input = training_datalist[0] # TODO...fake shape  for now
                self.build_graph(block_material, block_def, [local_input])
                block_material.local_graph = block_material.graph
                block_material.graph = None
            else:
                block_material.local_graph = None

            # discriminator
            input_images = training_datalist[0]
            self.build_graph(block_material, block_def, [input_images])
        
        except Exception as err:
            ezLogging.critical("%s - Build Graph; Failed: %s" % (block_material.id, err))
            block_material.dead = True
            #import pdb; pdb.set_trace()
            return

        block_material.output = [block_material.graph, block_material.local_graph]



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
        training_config_dict = training_datalist[-1]
        output_list = self.standard_evaluate(block_material, block_def, [training_config_dict])
        training_config_dict = output_list[0]
        block_material.output = [training_config_dict]
