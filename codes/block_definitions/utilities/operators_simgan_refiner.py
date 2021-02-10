'''
root/codes/block_definitions/utilities/operators...

Overview:
This file has operators for a SimGAN refiner (must maintain the shape of the input)

Rules:
The only big rule is that the refiner must maintain the shape of the input 
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
from codes.block_definitions.utilities.operators_pytorch import PyTorchLayerWrapper, linear_layer
from codes.utilities.custom_logging import ezLogging

### init dict
operator_dict = {}


def conv1d_layer(input_layer, misc_layer, out_channels, kernel_size=3, activation=nn.ReLU):
    """
    An operator that initializes a Conv1D_Layer_Refiner object, which holds a PyTorch Conv1d layer that maintains the input shape. Ignores the misc_layer.
    """
    class Conv1D_Layer_Refiner(PyTorchLayerWrapper):
        def __init__(self, in_shape, out_channels, kernel_size, activation):
            # Store information from args
            self.in_channels = in_shape[0]
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.padding = kernel_size//2
            self.activation = activation

            # Define out shape and instantiate layer
            num_dims = len(in_shape)
            self.out_shape = (out_channels, in_shape[-1])

            layers = []
            # Make sure we have 2-d shape for channels and signal length
            if len(in_shape) == 1:
                layers.append(nn.Unflatten(dim=0, unflattened_size=(1,)))
            layers.append(nn.Conv1d(self.in_channels, out_channels, kernel_size, padding=self.padding)) # Add conv layer
            # Add activation layer
            if activation is not None:
                layers.append(activation())
            self.layer = nn.Sequential(*layers)
    
    return Conv1D_Layer_Refiner(input_layer.get_out_shape(), out_channels, kernel_size, activation)


operator_dict[conv1d_layer] = {"inputs": [PyTorchLayerWrapper, PyTorchLayerWrapper],
                               "output": PyTorchLayerWrapper,
                               "args": [argument_types.ArgumentType_Pow2, argument_types.ArgumentType_PyTorchKernelSize, 
                                        argument_types.ArgumentType_PyTorchActivation]
                              }
