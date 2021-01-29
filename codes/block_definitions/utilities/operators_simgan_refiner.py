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

## Add methods from operators_pytorch
operator_dict[linear_layer] = {"inputs": [Tensor],
                               "output": Tensor,
                               "args": [argument_types.ArgumentType_Pow2]
                              }


def conv1d_layer(in_channels, out_channels, kernel_size=3, activation=nn.ReLU):
    class Conv1D_Layer_Refiner(PyTorchLayerWrapper):
        def __init__(self, in_channels, out_channels, kernel_size, activation):
            self.out_size = out_channels
            self.kernel_size = kernel_size
            self.padding = padding or kernel_size//2
            self.activation = activation
            if activation is not None:
                self.layer = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size, padding),
                        activation()
                    )
            else:
                self.layer = nn.Conv1d(in_channels, out_size, kernel_size, padding)
    
    return Conv1D_Layer(in_channels, out_channels, kernel_size, activation)

operator_dict[conv1d_layer] = {"inputs": [Tensor],
                               "output": Tensor,
                               "args": [argument_types.ArgumentType_Pow2, argument_types.ArgumentType_PyTorchKernelSize, 
                                        argument_types.ArgumentType_PyTorchActivation]
                              }
