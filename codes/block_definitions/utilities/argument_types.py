'''
root/code/block_definitions/utilities/argument_types.py

Overview:
Strongly typing the CGP and having a separate GA evolution for our arguments means it would be easiest to create individual classes for each of our argument data types but at least have all follow the same abstract class.
At the minimum, it needs a .value attribute to store the actual value of an instantiated argument class, and a method to mutate.
There is a lot of experimentation to be done to make sure that we have rhobust enough mutation methods: if I want a simple integer, but it needs to be a large value, is there any way to guarentee that ther will be an argument close enough?

Rules:
Basically only needs .value and mutate() defined.
'''

### packages
import numpy as np
from numpy import random as rnd
from copy import copy, deepcopy
from abc import ABC, abstractmethod
import logging

'''
### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
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


    '''
    Mutation Methods:
    Define a set of mutation methods to be called on to mutate all/any of the argument classes.
    Currently, the way we set the boundaries of the uniform or the std of the normal distribution are entirely arbitrary. No analysis has been done on this yet.
    
    There is a bias here in the code to try and encourage values to move away from negative values as you can see in the conditions if the value is 0...a lot needs to be changed with these methods.
    '''
    def mut_uniform(self):
        if self.value == 0:
            low = 0
            high = 5
        else:
            low = self.value*.85
            high = self.value * 1.15
        logging.debug("%s-%s - numpy.random.uniform(%f,%f)" % (None, None, low, high))
        self.value = rnd.uniform(low,high)
            

    def mut_normal(self):
        if self.value == 0:
            mean = 3
            std = 3*.1
        else:
            mean = self.value
            std = self.value * .1
        logging.debug("%s-%s - numpy.random.normal(%f,%f)" % (None, None, mean, std))    
        self.value = rnd.normal(mean, std)



class ArgumentType_Ints(ArgumentType_Abstract):
    '''
    To try and capture a large range of ints, 1/3 of ints will start at 5,
    another third will start at 50, and the final third will start at 100.
    Then all will mutate around that value.
    All ints are bounded by 1 such that [1,?)...after mutating, we force anything 
    less than 1, to 1.
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
            self.mutate()
        else:
            self.value = value
        logging.debug("%s-%s - Initialize ArgumentType_Ints Class to %f" % (None, None, self.value))


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
        logging.debug("%s-%s - Mutated ArgumentType_Ints to %f" % (None, None, self.value))



class ArgumentType_Pow2(ArgumentType_Abstract):
    '''
    This can be any number 2**i with i any int {1,2,3,4,5,6,7,8}
    
    Commonly used in CNN for setting the size of the convolutions.
    '''
    def __init__(self, value=None):
        if value is None:
            self.mutate()
        else:
            self.value = value
        logging.debug("%s-%s - Initialize ArgumentType_Pow2 Class to %f" % (None, None, self.value))


    def mutate(self):
        roll = rnd.random_integers(1, 9)
        self.value = int(2 ** roll)
        logging.debug("%s-%s - Mutated ArgumentType_Pow2 to %f" % (None, None, self.value))



class ArgumentType_SmallFloats(ArgumentType_Abstract):
    '''
    Here we get 'small' floats in that they are initialized with a normal
    distribution centered around 10 so it would be extremely unlikely for it
    to mutate to a number say 100+.
    '''
    def __init__(self, value=None):
        if value is None:
            self.value = rnd.random()*10
        else:
            self.value = value
        logging.debug("%s-%s - Initialize ArgumentType_SmallFloats Class to %f" % (None, None, self.value))


    def mutate(self):
        roll = rnd.random_integers(0,1)
        if roll == 0:
            self.mut_normal()
        elif roll == 1:
            self.mut_uniform()
        else:
            pass
        logging.debug("%s-%s - Mutated ArgumentType_SmallFloats to %f" % (None, None, self.value))