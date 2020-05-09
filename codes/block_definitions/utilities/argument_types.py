'''
root/code/block_definitions/utilities/argument_types.py

Overview:
    Define a set of mutation methods to be called on to mutate all/any of the argument classes.
    Consider limiting them to be strictly positive or non-negative

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
import numpy as np
from numpy import random as rnd
from copy import copy, deepcopy
from abc import ABC, abstractmethod

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root



### Argument Classes
'''
    Define the various argument classes with an __init__() and mutate() function.
    Make sure that there is also a self.value attribute, and a self.num_samples
    attribute to define how many samples of that argument object type to create
    in the argument skeleton (see below)
    class ArgumentType(object):
        def __init__(self):
            self.num_samples = #
            self.value = #######
            self.mutate()
        def mutate(self):
            roll = r.random_integers(0,#)
            if roll == 0:
                self.value = #######
            elif roll == 1:
                self.value = #######
            ...

    Once an Argument Class is defined, at it to the list of arguments.
    The argument skeleton is filled by sampling from this list so even if
    an Arg Class is defined but not added to the list, it will not be used.
'''
class ArgumentType_Abstract(ABC):
    @abstractmethod
    def __init__(self):
        pass


    @abstractmethod
    def mutate(self):
        pass


    def __str__(self):
        return "{}".format(self.value)


    def __repr__(self):
        return str(self)


    ### Mutation Methods
    '''
        Define a set of mutation methods to be called on to mutate all/any of the argument classes.
        Consider limiting them to be strictly positive or non-negative
    '''
    def mut_uniform(self):
        if self.value == 0:
            return rnd.uniform(0,5)
        else:
            low = self.value*.85
            high = self.value * 1.15
            self.value = rnd.uniform(low,high)


    def mut_normal(self):
        if self.value == 0:
            return rnd.normal(3, 3*.1)
        else:
            mean = self.value
            sd = self.value * .1
            self.value = rnd.normal(mean, sd)



class ArgumentType_Ints(ArgumentType_Abstract):
    '''
    TODO
    '''
    def __init__(self, value=None):
        if value is None:
            roll = rnd.random_integers(0,2)
            if roll == 0:
                self.value = 5
            elif roll == 1:
                self.value = 50
            elif roll == 2:
                self.value = 100
            self.num_samples = 10
            self.mutate()
        else:
            self.value = value
            self.num_samples = 10


    def mutate(self):
        roll = rnd.random_integers(0,1)
        if roll == 0:
            self.mut_normal()
        elif roll == 1:
            self.mut_uniform()
        else:
            pass
        if self.value < 1:
            self.value = 1
        else:
            pass
        self.value = int(self.value)



class ArgumentType_Pow2(ArgumentType_Abstract):
    '''
    TODO
    '''
    def __init__(self, value=None):
        if value is None:
            self.mutate()
        else:
            self.value = value
            self.num_samples = 10


    def mutate(self):
        roll = rnd.random_integers(1, 9)
        self.value = int(2 ** roll)