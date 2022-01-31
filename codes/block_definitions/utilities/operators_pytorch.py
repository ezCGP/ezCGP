
'''
root/codes/block_definitions/utilities/operators...

Overview:
This file has operators for PyTorch nueral network layers

Rules:
There are several unique features to building and then running a NN with PyTorch.
It is expected that you would instantiate various nn.Module objects in the __init__
method of your graph's class, and then show how those modules connect to each other
in the forward method.
So our block_evaluate.evaluate() method will define a class for our network and fills
in the __init__ and forward method appropriately.
To assist with that, the primitives need to all have the same rigid structure:
* primitive inputs:
    - a list of input layer shapes, each shape is a tuple (pytorch needs this for most of its modules)
    - remaining positional args to be passed to the module we want to create
* need some uniform way to get the shape of the tensor output by each primitive
* returns an instance of some nn.Module class OR of our custom MimicPyTorchModule
  class that turns methods into callable classes so that they look like nn.Module
  objects

expect data to be of shape -> (number_of_samples, num_channels=1, image_length=92)
...remember that it's just 1D data

TODO!!! need to go through and verify which hyperparameters we want to evolve for the primitives...
some are selected by me and the rest by anthony
'''


### packages
from abc import ABC, abstractmethod
import torch
from torch import nn
import numpy as np
import copy

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.block_definitions.utilities import argument_types
from codes.utilities.custom_logging import ezLogging

### init dict
operator_dict = {}



class MimicPyTorchModule(ABC):
    '''
    When trying to evaluate a graph with both torch.nn.Module class instances (ex torch.nn.Conv1d) and
    torch methods (ex torch.cat), there was a problem with knowing the input layer shapes if a
    torch method fed into a torch.nn.Module instance.
    So the idea is to wrap all our methods into callable classes so that they all operate like torch.nn.Modules
    and so we can force the user to define the output layer shape of the methods, so that if they are fed to
    torch.nn.Modules we would know what the input shape would be.

    ASSUMPTIONS
    To allow for any number of inputs, the input_shapes is expected to be a list of tuples with len number of
    inputs and where the tuples store each shape.
    Also going to assume that multiple inputs can be passed in like *args instead of as a seuqnece of inputs
    like args (dropped asterisk). If a sequence is needed, you're going to have to overwrite the __call__ method.

    '''
    def __init__(self, input_shapes, ftn, **kwargs):
        self.input_shapes = input_shapes
        self.ftn = ftn
        self.kwargs = kwargs

    @abstractmethod
    def get_out_shape(self):
        pass

    def __call__(self, *args):
        return self.ftn(*args, **self.kwargs)


def pytorch_squeeze(input_shapes, dim=None):
    class PyTorch_Squeeze(MimicPyTorchModule):
        '''
        https://pytorch.org/docs/stable/generated/torch.squeeze.html
        '''
        def __init__(self, input_shapes, dim=None):
            super().__init__(input_shapes, ftn=torch.squeeze, dim=dim)

        def __call__(self, *args):
            # for some reason, if dim is None, then it erros...someone should get fired
            del self.kwargs['dim']
            return super().__call__(*args)

        def get_out_shape(self):
            shape = list(self.input_shapes[0])

            drop_index = []
            if ('dim' not in self.kwargs) or (self.kwargs['dim'] is None):
                for index, value in enumerate(shape):
                    if value == 1:
                        drop_index.append(index)
            else:
                if shape[self.kwargs['dim']] == 1:
                    drop_index.append(self.kwargs['dim'])
            
            for index in reversed(drop_index):
                _  = shape.pop(index)

            return tuple(shape)

    return PyTorch_Squeeze(input_shapes, dim)


def pytorch_concat(input_shapes, dim=0):
    '''
    PROBLEM: originally was super clever to allow concat to work on any number of items,
    but then I forgot about how the operator_dict will force arity to be well defined...crap.
    I guess for now just assume 2 things being concat and that we can apply this method twice
    to concat 3 things.
    '''
    class PyTorch_Concat(MimicPyTorchModule):
        '''
        https://pytorch.org/docs/stable/generated/torch.cat.html

        Tested with:
            x = torch.randn(2,3,2)
            y = torch.randn(3,3,2)
            ting = PyTorch_Concat(input_shapes = [x.shape, y.shape], dim = 6)
            z = ting(tensors=[x,y])
            print(ting.get_out_shape(), z.shape)
        '''
        def __init__(self, input_shapes, dim=0):
            # as an attempt to coerce data to not error on concate if number of dimensions differ...
            # check numer of dimensions and flatten to smallest
            num_dims = []
            for shape in input_shapes:
                num_dims.append(len(shape))
            num_dims = np.array(num_dims)
            smallest_dim = num_dims.min()
            self.flatten_ith_shape = []
            if num_dims.var() != 0.0:
                for i, shape in enumerate(input_shapes):
                    if len(shape) > smallest_dim:
                        self.flatten_ith_shape.append(i)
                        input_shapes[i] = (shape[0], np.array(shape)[1:].prod())

            # going to assume input_shapes have the same number of dimensions; if not it will error in __call__
            dim = dim % smallest_dim
            super().__init__(input_shapes, ftn=torch.cat, dim=dim)

        def __call__(self, *args):
            args = list(args)
            # complete coercion if num of dims in data is different
            for i, val in enumerate(args):
                args[i] = val.flatten(start_dim=1, end_dim=-1)

            # this method takes in a SEQUENCE of tensors as a single input, so dropping the asterisk
            return super().__call__(args)

        def get_out_shape(self):
            '''
            going to assume that the shapes of the things we want to concat are valid.
            also assuming we will never concat the 0th dim since that's the size of the batch
            
            this code is weird and the behavior of concat is weird...sometimes the other dimensions
            get flattened instead of maintaining the number of dimensions...wtf
            
            shape = ()
            for dim in range(len(self.input_shapes[0])):
                length = 0
                for input_shape in self.input_shapes:
                    length += input_shape[dim]
                    if dim != self.kwargs['dim']:
                        break
                shape += (length,)
            '''
            shape = (None,)
            for dim in range(1, len(self.input_shapes[0])):
                if dim == self.kwargs['dim']:
                    # then this is the dim we are concatenating
                    length = 0
                    for tensor_shape in self.input_shapes:
                        length += int(tensor_shape[dim])
                    shape += (length,)
                else:
                    shape += (self.input_shapes[0][dim])
            return shape

    return PyTorch_Concat(input_shapes, dim)

operator_dict[pytorch_concat] = {"inputs": [nn.Module, nn.Module],
                                 "output": nn.Module,
                                 "args": [argument_types.ArgumentType_Int0to25]
                                }



##### NN.MODULE #####
class WrapPyTorchModule(ABC):
    def __init__(self, input_shapes, **kwargs):
        self.input_shapes = input_shapes
        for key, value in kwargs.items():
            self.__dict__[key] = value

    @abstractmethod
    def get_out_shape(self):
        pass


def linear_layer(input_shapes, *args):
    '''
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear

    TODO consider adding activation arg similar to conv1d
    '''
    class PyTorch_Linear(WrapPyTorchModule, nn.Linear):
        def __init__(self, input_shapes, out_features):
            WrapPyTorchModule.__init__(self, input_shapes)
            assert(len(input_shapes)==1), "expected 1 input but got %i" % len(input_shapes)
            in_features = input_shapes[0][-1] # get number of channels
            nn.Linear.__init__(self, in_features, out_features, bias=True)

        def get_out_shape(self):
            out_features = self.out_features
            output_shape = self.input_shapes[0][:-1] + (out_features,)
            return output_shape

    return PyTorch_Linear(input_shapes, *args)

operator_dict[linear_layer] = {"inputs": [nn.Module],
                               "output": nn.Module,
                               "args": [argument_types.ArgumentType_Pow2]
                              }


def conv1d_layer(input_shapes, *args):
    '''
    https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d

    TODO!!!!! ADD STRIDE AND THEN ADJUST GET_OUT_sHAPE
    '''
    class PyTorch_Conv1d(WrapPyTorchModule, nn.Sequential):
        def __init__(self, input_shapes, out_channels, kernel_size=3, stride=1, padding=0, activation=nn.ReLU()):
            WrapPyTorchModule.__init__(self, input_shapes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

            in_channels = input_shapes[0][-2] # get number of channels
            modules = []
            modules.append(nn.Conv1d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=1,
                                     groups=1,
                                     bias=True,
                                     padding_mode='zeros'))

            if activation is not None:
                modules.append(activation)

            nn.Sequential.__init__(self, *modules)

        def get_out_shape(self):
            out_channels = self.out_channels
            # going to assume 3 dimensions in shape
            num_samples, num_channels, sample_length = self.input_shapes[0]
            #new_sample_length = sample_length - (self.kernel_size-1) + self.padding*2
            new_sample_length = (sample_length - self.kernel_size + 2*self.padding)//self.stride + 1
            output_shape = (num_samples, out_channels, new_sample_length)
            return output_shape

    return PyTorch_Conv1d(input_shapes, *args)

operator_dict[conv1d_layer] = {"inputs": [nn.Module],
                               "output": nn.Module,
                               "args": [argument_types.ArgumentType_Pow2,
                                        argument_types.ArgumentType_PyTorchKernelSize,
                                        argument_types.ArgumentType_PyTorchStrideSize,
                                        argument_types.ArgumentType_PyTorchPaddingSize,
                                        argument_types.ArgumentType_PyTorchActivation]
                              }


def batch_normalization(input_shapes, *args):
    '''
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d
    '''
    class PyTorch_BatchNorm1d(WrapPyTorchModule, nn.BatchNorm1d):
        def __init__(self, input_shapes, eps=1e-05, momentum=0.1, affine=True):
            WrapPyTorchModule.__init__(self, input_shapes, eps=eps, momentum=momentum, affine=affine)

            if len(input_shapes[0]) == 2:
                num_features = input_shapes[0][-1]
            elif len(input_shapes[0]) > 2:
                num_features = input_shapes[0][-2]
            else:
                # no idea...should error
                num_features = 0

            nn.BatchNorm1d.__init__(self, num_features, eps, momentum, affine)

        def get_out_shape(self):
            # shape shouldn't change
            return self.input_shapes[0]

    return PyTorch_BatchNorm1d(input_shapes, *args)

operator_dict[batch_normalization] = {"inputs": [nn.Module],
                                      "output": nn.Module,
                                      "args": [] #TODO
                                     }


def flatten_layer(input_shapes, *args):
    '''
    https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html#torch.nn.Flatten
    '''
    class PyTorch_Flatten(WrapPyTorchModule, nn.Flatten):
        def __init__(self, input_shapes, start_dim=1, end_dim=-1):
            WrapPyTorchModule.__init__(self, input_shapes, start_dim=start_dim, end_dim=end_dim)
            nn.Flatten.__init__(self, start_dim, end_dim)

        def get_out_shape(self):
            # assume 1 input
            start_shape = tuple(self.input_shapes[0][:self.start_dim])
            if self.end_dim==-1:
                flattened_shape = (np.array(self.input_shapes[0][self.start_dim:]).prod(),)
                end_shape = ()
            else:
                flattened_shape = (np.array(self.input_shapes[0][self.start_dim:self.end_dim+1]).prod(),)
                end_shape = tuple(self.input_shapes[0][self.end_dim+1:])

            return start_shape + flattened_shape + end_shape

    return PyTorch_Flatten(input_shapes, *args)

operator_dict[flatten_layer] = {"inputs": [nn.Module],
                                "output": nn.Module,
                                "args": []
                               }


def softmax_layer(input_shapes, *args):
    '''
    no operator_dict entry yet so it isn't used in evolution
    https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
    '''
    class PyTorch_Softmax(WrapPyTorchModule, nn.Softmax):
        def __init__(self, input_shapes):
            WrapPyTorchModule.__init__(self, input_shapes)
            nn.Softmax.__init__(self, dim=1)

        def get_out_shape(self):
            return self.input_shapes[0]

    return PyTorch_Softmax(input_shapes, *args)


def sigmoid_layer(input_shapes, *args):
    '''
    no operator_dict entry yet so it isn't used in evolution
    https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
    '''
    class PyTorch_Sigmoid(WrapPyTorchModule, nn.Sigmoid):
        def __init__(self, input_shapes):
            WrapPyTorchModule.__init__(self, input_shapes)
            nn.Sigmoid.__init__(self)

        def get_out_shape(self):
            return self.input_shapes[0]

    return PyTorch_Sigmoid(input_shapes, *args)


def avg_pool(input_shapes, *args):
    '''
    https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html#torch.nn.AvgPool1d
    '''
    class PyTorch_AvgPool1d(WrapPyTorchModule, nn.AvgPool1d):
        def __init__(self, input_shapes, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
            if stride is None:
                stride = kernel_size
            WrapPyTorchModule.__init__(self, input_shapes, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)
            nn.AvgPool1d.__init__(self, kernel_size, stride, padding, ceil_mode, count_include_pad)

        def get_out_shape(self):
            # going to assume 3 dimensions in shape
            num_samples, num_channels, sample_length = self.input_shapes[0]
            # nn.AvgPool1d sets stride, kernel_size, padding attributes and overwrites the WrapPyTorchModule ones as tuples
            # so have to add in the [0] for those
            new_sample_length = (sample_length - self.kernel_size[0] + 2*self.padding[0])//self.stride[0] + 1
            output_shape = (num_samples, num_channels, new_sample_length)
            return output_shape

    return PyTorch_AvgPool1d(input_shapes, *args)

operator_dict[avg_pool] = {"inputs": [nn.Module],
                           "output": nn.Module,
                           "args": [argument_types.ArgumentType_PyTorchKernelSize,
                                    argument_types.ArgumentType_PyTorchStrideSize,
                                    argument_types.ArgumentType_PyTorchPaddingSize]
                          }


def max_pool(input_shapes, *args):
    '''
    https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html#torch.nn.MaxPool1d
    '''
    class PyTorch_MaxPool1d(WrapPyTorchModule, nn.MaxPool1d):
        def __init__(self, input_shapes, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
            if stride is None:
                stride = kernel_size
            WrapPyTorchModule.__init__(self, input_shapes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
            nn.MaxPool1d.__init__(self, kernel_size, stride, padding, dilation, return_indices, ceil_mode)

        def get_out_shape(self):
            features = self.input_shapes[0][-1]
            return (features - self.kernel_size + 2*self.padding)//self.stride + 1

    return PyTorch_MaxPool1d(input_shapes, *args)

operator_dict[max_pool] = {"inputs": [nn.Module],
                           "output": nn.Module,
                           "args": [argument_types.ArgumentType_PyTorchKernelSize] #TODO
                          }


def dropout(input_shapes, *args):
    '''
    https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout
    '''
    class Dropout_Layer(WrapPyTorchModule, nn.Dropout):
        def __init__(self, out_shape, p=0.5):
            WrapPyTorchModule.__init__(self, input_shapes, p=p)
            nn.Dropout.__init__(self, p)

        def get_out_shape(self):
            return self.input_shapes[0]

    return Dropout_Layer(input_shapes, *args)

operator_dict[dropout] = {"inputs": [nn.Module],
                          "output": nn.Module,
                          "args": [argument_types.ArgumentType_LimitedFloat0to1] #TODO
                         }


def resnet(input_shapes, nb_channels=16, kernel_size=5, use_leaky_relu=True):
    '''
    https://eoslcm.gtri.gatech.edu/bitbucket/projects/EMADE/repos/simgan/browse/network.py?at=2021-dev#9
    '''
    class Resnet(nn.Module):
        '''
        PROBLEM: for certain sizes of kernel_size (even?), the output num features won't match the original.
        Also, how is the 'adding' going to work with x having in_channels and convs having nb_channels?
        '''
        def __init__(self, in_channels, nb_channels, kernel_size, use_leaky_relu):
            super().__init__()
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



    class ResnetBlock(WrapPyTorchModule, Resnet):
        def __init__(self, input_shapes, nb_channels, kernel_size, use_leaky_relu):
            in_channels = input_shapes[0][1]
            WrapPyTorchModule.__init__(self, input_shapes, in_channels=in_channels, nb_channels=nb_channels, kernel_size=kernel_size, use_leaky_relu=use_leaky_relu)
            Resnet.__init__(self, in_channels, nb_channels, kernel_size, use_leaky_relu)

        def get_out_shape(self):
            '''
            i guess i can do a series of calculations for each layer but what happens if x and mysum are different shapes because of padding?
            '''
            # TODO!!!!!
            return self.input_shapes[0]

    return ResnetBlock(input_shapes, nb_channels, kernel_size, use_leaky_relu)

operator_dict[resnet] = {"inputs": [nn.Module],
                         "output": nn.Module,
                         "args": [argument_types.ArgumentType_Pow2,
                                  argument_types.ArgumentType_PyTorchKernelSize,
                                  argument_types.ArgumentType_Bool] #TODO
                        }


def minibatch_discrimination(input_shapes, *args):
    '''
    stuff
    '''
    class MinibatchDiscrimination(WrapPyTorchModule, nn.Module):
        def __init__(self, input_shapes, out_features, kernel_dims, mean=False):
            in_features = input_shapes[0][1] # NOTE: assumed images are already flattened
            WrapPyTorchModule.__init__(self, input_shapes, in_features=in_features, out_features=out_features, kernel_dims=kernel_dims, mean=mean)
            nn.Module.__init__(self)

            self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
            nn.init.normal_(self.T, mean=0, std=1)

        def forward(self, x):
            '''
            x -> N x (in_features) ...assumed x already flattened
            T -> (in_features) x (out_features) x (kernel)
            '''
            T_reshaped = self.T.view(self.in_features, -1)                    # (in_features) x (out_features*kernel)
            matrices = x.mm(T_reshaped)                                       # N x (out_features*kernel)
            matrices = matrices.view(-1, self.out_features, self.kernel_dims) # N x (out_features) x (kernel)
            M = matrices.unsqueeze(0)                                         # 1 x N x (out_features) x (kernel)
            M_T = M.permute(1, 0, 2, 3)                                       # N x 1 x (out_features) x (kernel)
            norm = torch.abs(M-M_T).sum(3)                                    # N x N x (out_features)
            expnorm = torch.exp(-norm)                                        # N x N x (out_features)
            o_b = expnorm.sum(0)                                              # N x (out_features)
            return o_b

        def get_out_shape(self):
            N = self.input_shapes[0][0]
            return (N, self.out_features)

    return MinibatchDiscrimination(input_shapes, *args)

operator_dict[minibatch_discrimination] = {"inputs": [nn.Module],
                                           "output": nn.Module,
                                           "args": [argument_types.ArgumentType_Pow2,
                                                    argument_types.ArgumentType_PyTorchKernelSize] #TODO
                                          }



def feature_extraction(input_shapes, *args):
    '''
    just calls some feature extraction class that was stolen from original simgan codebase.
    should be in misc folder

    NOTE: an idea was to make the output of this method to be a 2nd input into the 
    discriminator block instead of a primitive of that block
    '''
    from misc import features
    class FeatureExtractor(MimicPyTorchModule, features.FeatureExtractor):
        def __init__(self, input_shapes, *args):
            MimicPyTorchModule.__init__(self, input_shapes, ftn=None)
            features.FeatureExtractor.__init__(self, *args)

        def get_out_shape(self):
            return (self.input_shapes[0][0], self.num_features)

        def __call__(self, x):
            # not sure why i have to do all this .cpu() etc stuff but in original simgan code
            features = self.get_features(np.squeeze(x.cpu().detach().numpy()))
            this = torch.Tensor(np.transpose(features)).to(x.device)
            return this

    return FeatureExtractor(input_shapes, *args)

operator_dict[feature_extraction] = {"inputs": [nn.Module],
                                     "output": nn.Module,
                                     "args": [argument_types.ArgumentType_Bool]*9
                                     }
