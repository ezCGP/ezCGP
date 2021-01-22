'''
root/codes/block_definitions/utilities/operators...

Overview:
This file has operators for PyTorch nueral network layers

Rules:
Because PyTorch doesn't have a compile function for neural networks, you must know the shape of the input for certain layers, such as Conv1D where you need to know the input_channels. In order to solve that, we add a layer wrapper that stores the arguments and then instantiates the layer when we have the entire structure of the network. That way we will know the 
'''


### packages
from abc import ABC, abstractmethod
from torch import nn
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


    @abstractmethod
    def init_layer(self):
        pass


def pooling_layer():
    pass


def linear_layer(input_tensor, out_features):
    class Linear_Layer(PyTorchLayerWrapper):
        def __init__(self, out_features):
            self.out_features = out_features
        
        def init_layer(self, in_features):
            return nn.Linear(in_features, self.out_features)

    return Linear_Layer(out_features)

operator_dict[linear_layer] = {"inputs": [nn],
                               "output": nn,
                               "args": [argument_types.ArgumentType_Pow2]
                              }


def conv1d_layer(input_tensor, out_channels, kernel_size=3, padding=None, activation=nn.ReLU):
    class Conv1D_Layer(PyTorchLayerWrapper):
        def __init__(self, out_channels, kernel_size, padding, activation):
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.padding = padding or kernel_size - 1 # if padding is none, automatically match kernel_size-1 to maintain shape
            self.activation = activation

        def init_layer(self, in_channels):
            if activation is not None:
                return nn.Sequential(
                    nn.Conv1D(in_channels, self.out_channels, kernel_size, padding),
                    self.activation
                )
            return nn.Conv1D(in_channels, self.out_channels, kernel_size, padding)
    
    return Conv1D_Layer(out_channels, kernel_size, padding, activation)

operator_dict[conv1d_layer] = {"inputs": [nn],
                               "output": nn,
                               "args": [argument_types.ArgumentType_Pow2, argument_types.ArgumentType_PyTorchKernelSize,
                                        argument_types.ArgumentType_PyTorchPaddingSize, argument_types.ArgumentType_PyTorchActivation]
                              }
