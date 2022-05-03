'''
play with torch networks...

I know we'll want to be able to make a non-sequential graph, so how
can I be clever about making this graph and adding layers in __init__()
and using them in forward()?
'''

import math
import inspect
import torch
from torch import nn


### Build a genome

genome = [None]*7
genome[0] = {"function": torch.add,
             "inputs": [-1, -1],
             "args": []}

genome[1] = {"function": nn.Linear,
             "inputs": [-1],
             "args": [10, 5]}

genome[2] = {"function": nn.Linear,
             "inputs": [0],
             "args": [10, 5]}


genome[3] = {"function": nn.Bilinear,
             "inputs": [1, 2],
             "args": [5, 5, 4]}


genome[4] = {"function": nn.Linear,
             "inputs": [3],
             "args": [4, 3]}

genome[5] = 4


### format what it would look like to create a custom pytorch network with genome
class MyNetwork(nn.Module):
    '''
    model.parameters() is empty unless I call add_module() better way to do it?
    '''
    def __init__(self, genome):
        super().__init__()
        self.node_inputs = {}

        for node_index in range(5):
            function = genome[node_index]['function']
            args = genome[node_index]['args']
            self.node_inputs[node_index] = genome[node_index]['inputs']
            if inspect.isclass(function):
                self.add_module(name='node_%i' % node_index,
                                module=function(*args))
            else:
                # assume it is a method and not some nn.Module then
                self.__dict__['node_%i' % node_index] = {'function': function,
                                                         'args': args}


    def forward(self, x):
        self.evaluated = [None]*len(self.node_inputs)

        for node_index in range(5):
            node_str = 'node_%i' % node_index
            if node_str in self._modules:
                function = self._modules[node_str]
                args = []
            else:
                function = self.__dict__[node_str]['function']
                args = self.__dict__[node_str]['args']

            input_index = self.node_inputs[node_index]
            inputs = []
            for index in input_index:
                if index < 0:
                    inputs.append(x)
                else:
                    inputs.append(self.evaluated[index])

            self.evaluated[node_index] = function(*inputs, *args)

        return self.evaluated[4]


### make fake data
X = torch.rand(30, 10)
y = torch.rand(30, 3)

model = MyNetwork(genome)
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.RMSprop(model.parameters(),
                                lr=1e-3)
print(model._modules['node_1'].bias) # value before training
y_pred = model(X)
loss = loss_fn(y_pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(model._modules['node_1'].bias) # value after training
