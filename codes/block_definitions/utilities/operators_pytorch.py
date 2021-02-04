
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


def pooling_layer():
    pass


def linear_layer(in_shape, out_features):
    class Linear_Layer(PyTorchLayerWrapper):
        def __init__(self, in_shape, out_features):
            # Store information from args
            self.in_features = 1
            for dim in list(in_shape): 
                self.in_features *= dim
            self.out_features = out_features

            # Define out shape and instantiate layer
            self.out_shape = (out_features,)
            self.layer = nn.Sequential(
                    nn.Flatten(start_dim=1), # gives us an NxD tensor (ideal for linear layer)
                    nn.Linear(self.in_features, out_features)
                )
        
    return Linear_Layer(in_shape, out_features)

operator_dict[linear_layer] = {"inputs": [Tensor, Tensor],
                               "output": Tensor,
                               "args": [argument_types.ArgumentType_Pow2]
                              }


def conv1d_layer(in_shape, out_channels, kernel_size=3, padding=None, activation=nn.ReLU):
    class Conv1D_Layer(PyTorchLayerWrapper):
        def __init__(self, in_shape, out_channels, kernel_size, padding, activation):
            # Store information from args
            self.in_channels = in_shape[0]
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.padding = padding or kernel_size//2 # if padding is none, automatically match kernel_size//2 to maintain shape
            self.activation = activation

            # Define out shape and instantiate layer
            num_dims = len(in_shape)
            self.out_shape = (out_channels, in_shape[-1])

            layers = []
            # Make sure we have 2-d shape for channels and signal length
            if len(in_shape) == 1:
                layers.append(nn.Unflatten(dim=0, unflattened_size=(1,)))
            layers.append(nn.Conv1d(self.in_channels, out_channels, kernel_size, padding=padding)) # Add conv layer
            # Add activation
            if activation is not None:
                layers.append(activation())
            self.layer = nn.Sequential(*layers)
    
    return Conv1D_Layer(in_shape, out_channels, kernel_size, padding, activation)

operator_dict[conv1d_layer] = {"inputs": [Tensor, Tensor],
                               "output": Tensor,
                               "args": [argument_types.ArgumentType_Pow2, argument_types.ArgumentType_PyTorchKernelSize,
                                        argument_types.ArgumentType_PyTorchPaddingSize, argument_types.ArgumentType_PyTorchActivation]
                              }