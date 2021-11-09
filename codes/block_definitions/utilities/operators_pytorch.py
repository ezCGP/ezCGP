
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
                               "args": [argument_types.ArgumentType_Pow2,
                                        argument_types.ArgumentType_PyTorchKernelSize,
                                        argument_types.ArgumentType_PyTorchPaddingSize,
                                        argument_types.ArgumentType_PyTorchActivation]
                              }


def batch_normalization(input_layer, eps=1e-5, momentum=0.1, affine=True):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d
    """
    class BatchNorm1d_Layer(PyTorchLayerWrapper):
        def __init__(self, num_features, eps, momentum, affine):
            self.out_shape = num_features # TODO, just guessing
            self.layer = nn.BatchNorm1d(num_features, eps, momentum, affine)

    return BatchNorm1d_Layer(input_layer.get_out_shape(), eps, momentum, affine)

operator_dict[batch_normalization] = {"inputs": [PyTorchLayerWrapper],
                                      "output": PyTorchLayerWrapper,
                                      "args": [] #TODO
                                     }


def avg_pool(input_layer, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html#torch.nn.AvgPool1d
    """
    class AvgPool1d_Layer(PyTorchLayerWrapper):
        def __init__(self, out_shape, kernel_size, stride, padding, ceil_mode, count_include_pad):
            self.out_shape = out_shape
            self.layer = nn.AvgPool1d(kernel_size, stride, padding, ceil_mode, count_include_pad)

    return AvgPool1d_Layer(input_layer.get_out_shape(), kernel_size, stride, padding, ceil_mode, count_include_pad)

operator_dict[avg_pool] = {"inputs": [PyTorchLayerWrapper],
                           "output": PyTorchLayerWrapper,
                           "args": [argument_types.ArgumentType_PyTorchKernelSize] #TODO
                          }


def max_pool(input_layer, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html#torch.nn.MaxPool1d
    """
    class MaxPool1d_Layer(PyTorchLayerWrapper):
        def __init__(self, out_shape, kernel_size, stride, padding, dilation, return_indices, ceil_mode):
            self.out_shape = out_shape
            self.layer = nn.MaxPool1d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    return MaxPool1d_Layer(input_layer.get_out_shape(), kernel_size, stride, padding, dilation, return_indices, ceil_mode)

operator_dict[max_pool] = {"inputs": [PyTorchLayerWrapper],
                           "output": PyTorchLayerWrapper,
                           "args": [argument_types.ArgumentType_PyTorchKernelSize] #TODO
                          }


def dropout(input_layer, p=0.5):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout
    """
    class Dropout_Layer(PyTorchLayerWrapper):
        def __init__(self, out_shape, p):
            self.out_shape = out_shape
            self.layer = nn.Dropout(p)

    return Dropout_Layer(input_layer.get_out_shape(), p)

operator_dict[dropout] = {"inputs": [PyTorchLayerWrapper],
                          "output": PyTorchLayerWrapper,
                          "args": [] #TODO
                         }


def resnet(input_layer, nb_channels=16, kernel_size=5, use_leaky_relu=True):
    """
    https://eoslcm.gtri.gatech.edu/bitbucket/projects/EMADE/repos/simgan/browse/network.py?at=2021-dev#9
    """
    class ResnetBlock(nn.Module):
        def __init__(self, in_channels, nb_channels, kernel_size, use_leaky_relu):
            super(ResnetBlock).__init__()
            self.convs = nn.Sequential(nn.Conv1d(in_channels, nb_channels, kernel_size, padding=kernel_size//2),
                                       nn.BatchNorm1d(nb_channels),
                                       nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
                                       nn.Conv1d(nb_channels, nb_channels, kernel_size, padding=kernel_size//2),
                                       nn.BatchNorm1d(nb_channels))
            self.relu = nn.ReLU()

        def forward(self, x):
            convs = self.convs(x)
            mysum = convs + x
            output = self.relu(mysum)
            return output

    return ResnetBlock(input_layer.get_out_shape(), nb_channels, kernel_size, use_leaky_relu)

operator_dict[resent] = {"inputs": [PyTorchLayerWrapper],
                         "output": PyTorchLayerWrapper,
                         "args": [] #TODO
                        }