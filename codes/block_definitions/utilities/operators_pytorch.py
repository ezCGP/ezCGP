
'''
root/codes/block_definitions/utilities/operators...

Overview:
This file has operators for PyTorch nueral network layers

Rules:
Because PyTorch doesn't have a compile function for neural networks, you must know the shape of the input for certain layers, such as Conv1D where you need to know the input_channels. In order to solve that, we add a layer wrapper that stores the arguments and then instantiates the layer when we have the entire structure of the network.
'''


### packages
from abc import ABC, abstractmethod
from torch import nn, Tensor
import numpy as np
from copy import deepcopy

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.block_definitions.utilities import argument_types
from codes.utilities.custom_logging import ezLogging

### init dict
operator_dict = {}

class PyTorchLayerWrapper():
    @abstractmethod
    def __init__(self):
        pass
    
    def get_out_shape(self):
        return self.out_shape

    def get_layer(self):
        return self.layer

class InputLayer(PyTorchLayerWrapper):
    def __init__(self, data):
        self.data = data
        self.out_shape = data.shape
        self.layer = None


def pooling_layer():
    pass

def linear_layer(input_layer, out_features):
    """
    An operator that initializes a Linear_Layer object, which holds a PyTorch Linear layer.
    """
    class Linear_Layer(PyTorchLayerWrapper):
        def __init__(self, in_shape, out_features):
            # Store information from args
            self.in_features = 1
            for dim in list(in_shape): 
                self.in_features *= dim
            self.out_features = out_features

            if self.out_features < 0:
                import pdb; pdb.set_trace()

            # Define out shape and instantiate layer
            num_dims = len(in_shape)
            self.out_shape = (out_features,)

            layers = []
            # Make sure we have 1-d shape
            if len(in_shape) == 2:
                layers.append(nn.Flatten(start_dim=1)) # gives us an NxD tensor (ideal for linear layer)

            layers.append(nn.Linear(self.in_features, self.out_features))
            
            self.layer = nn.Sequential(*layers)

    return Linear_Layer(input_layer.get_out_shape(), out_features)

operator_dict[linear_layer] = {"inputs": [PyTorchLayerWrapper],
                               "output": PyTorchLayerWrapper,
                               "args": [argument_types.ArgumentType_Pow2]
                              }


def conv1d_layer(input_layer, out_channels, kernel_size=3, padding=None, activation=nn.ReLU):
    """
    An operator that initializes a Conv1D_Layer object, which holds a PyTorch Conv1d layer.
    """
    class Conv1D_Layer(PyTorchLayerWrapper):
        def __init__(self, in_shape, out_channels, kernel_size, padding, activation):
            # In shape is the out shape of the last layer. NOTE: This doesn't include the batch dimension!
            # Store information from args
            self.out_channels = out_channels
            if self.out_channels < 0:
                import pdb; pdb.set_trace()
            self.kernel_size = kernel_size

            # if padding is none, automatically match kernel_size//2 to maintain shape
            if padding is None or padding == -1:
                self.padding = kernel_size//2
            else:
                self.padding = padding 
            self.activation = activation

            # Define out shape and instantiate layer
            num_dims = len(in_shape)
            self.out_shape = (out_channels, in_shape[-1] - (self.kernel_size - 1) + self.padding*2)

            layers = []
            # Make sure we have 2-d shape for channels and signal length
            if len(in_shape) == 1:
                layers.append(nn.Unflatten(dim=1, unflattened_size=(1, in_shape[0]))) # Converts a (D,) shape to (1, D) shape
                self.in_channels = 1
                layers.append(nn.Conv1d(self.in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding)) # Add conv layer
            else:
                self.in_channels = in_shape[0]
                layers.append(nn.Conv1d(self.in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding)) # Add conv layer

            # Add activation
            if activation is not None:
                layers.append(activation())
            self.layer = nn.Sequential(*layers)
    
    return Conv1D_Layer(input_layer.get_out_shape(), out_channels, kernel_size, padding, activation)

operator_dict[conv1d_layer] = {"inputs": [PyTorchLayerWrapper],
                               "output": PyTorchLayerWrapper,
                               "args": [argument_types.ArgumentType_Pow2, argument_types.ArgumentType_PyTorchKernelSize,
                                        argument_types.ArgumentType_PyTorchPaddingSize, argument_types.ArgumentType_PyTorchActivation]
                              }