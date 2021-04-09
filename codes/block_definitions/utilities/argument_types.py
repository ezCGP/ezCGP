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


### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.utilities.custom_logging import ezLogging



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
        ezLogging.debug("%s-%s - numpy.random.uniform(%f,%f)" % (None, None, low, high))
        self.value = rnd.uniform(low,high)
            

    def mut_normal(self):
        if self.value == 0:
            mean = 3
            std = 3*.1
        else:
            mean = self.value
            std = self.value * .1
        ezLogging.debug("%s-%s - numpy.random.normal(%f,%f)" % (None, None, mean, std))    
        self.value = rnd.normal(mean, std)



class ArgumentType_Bool(ArgumentType_Abstract):
    '''
    just your basic bool. mutate always switches to opposite value
    '''
    def __init__(self, value=None):
        if value is None:
            self.value = bool(np.random.choice([True,False]))
        else:
            self.value = bool(value)
        ezLogging.debug("%s-%s - Initialize ArgumentType_Bool Class to %f" % (None, None, self.value))


    def mutate(self):
        self.value = not self.value
        ezLogging.debug("%s-%s - Mutated ArgumentType_Bool to %f" % (None, None, self.value))



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
        ezLogging.debug("%s-%s - Initialize ArgumentType_Ints Class to %f" % (None, None, self.value))


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
        ezLogging.debug("%s-%s - Mutated ArgumentType_Ints to %f" % (None, None, self.value))



class ArgumentType_Pow2(ArgumentType_Abstract):
    '''
    This can be any number 2**i with i any int {1,2,3,4,5,6,7,8}
    
    Commonly used in CNN for setting the size of the convolutions.
    '''
    def __init__(self, value=None):
        if value is None:
            self.value = None
            self.mutate()
        else:
            self.value = value
        ezLogging.debug("%s-%s - Initialize ArgumentType_Pow2 Class to %f" % (None, None, self.value))


    def mutate(self):
        #choices = rnd.random_integers(1, 8)
        choices = list(np.arange(1,8+1))
        if self.value in choices:
            choices.remove(self.value) # works in-place
        pow2 = np.random.choice(choices)
        self.value = int(2**pow2)
        ezLogging.debug("%s-%s - Mutated ArgumentType_Pow2 to %f" % (None, None, self.value))



class ArgumentType_TFActivation(ArgumentType_Abstract):
    '''
    possible values:
    https://www.tensorflow.org/api_docs/python/tf/keras/activations

    returns the actual function
    '''
    def __init__(self, value=None):
        if value is None:
            self.value = None
            self.mutate()
        else:
            self.value = value
            self.get_name()
        ezLogging.debug("%s-%s - Initialize ArgumentType_TFActivation Class to %s" % (None, None, self.name))


    def get_name(self):
        if self.value is None:
            self.name = "None"
        else:
            self.name = self.value.__qualname__


    def mutate(self):
        import tensorflow as tf
        choices = [tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh, tf.nn.elu, None]
        if self.value in choices:
            choices.remove(self.value) # works in-place
        self.value = np.random.choice(choices)
        self.get_name()
        ezLogging.debug("%s-%s - Mutated ArgumentType_TFActivation to %s" % (None, None, self.name))



class ArgumentType_TFFilterSize(ArgumentType_Abstract):
    '''
    quick way to pick [1,3,5,7]
    '''
    def __init__(self, value=None):
        if value is None:
            self.value = None
            self.mutate()
        else:
            self.value = value
        ezLogging.debug("%s-%s - Initialize ArgumentType_TFFilterSize Class to %f" % (None, None, self.value))


    def mutate(self):
        choices = [1,3,5,7]
        if self.value in choices:
            choices.remove(self.value) # works in-place
        self.value = np.random.choice(choices)
        ezLogging.debug("%s-%s - Mutated ArgumentType_TFFilterSize to %f" % (None, None, self.value))



class ArgumentType_FilterSize(ArgumentType_Abstract):
    '''
    quick way to pick [3,5,7]
    '''
    def __init__(self, value=None):
        if value is None:
            self.value = None
            self.mutate()
        else:
            self.value = value
        ezLogging.debug("%s-%s - Initialize ArgumentType_FilterSize Class to %f" % (None, None, self.value))


    def mutate(self):
        choices = [3,5,7]
        if self.value in choices:
            choices.remove(self.value) # works in-place
        self.value = np.random.choice(choices)
        ezLogging.debug("%s-%s - Mutated ArgumentType_FilterSize to %f" % (None, None, self.value))



class ArgumentType_TFPoolSize(ArgumentType_Abstract):
    '''
    quick way to pick [1,2,3,4]
    '''
    def __init__(self, value=None):
        if value is None:
            self.value = None
            self.mutate()
        else:
            self.value = value
        ezLogging.debug("%s-%s - Initialize ArgumentType_TFPoolSize Class to %f" % (None, None, self.value))


    def mutate(self):
        choices = [1,2,3,4]
        if self.value in choices:
            choices.remove(self.value) # works in-place
        self.value = np.random.choice(choices)
        ezLogging.debug("%s-%s - Mutated ArgumentType_TFPoolSize to %f" % (None, None, self.value))



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
        ezLogging.debug("%s-%s - Initialize ArgumentType_SmallFloats Class to %f" % (None, None, self.value))


    def mutate(self):
        roll = rnd.random_integers(0,1)
        if roll == 0:
            self.mut_normal()
        elif roll == 1:
            self.mut_uniform()
        else:
            pass
        ezLogging.debug("%s-%s - Mutated ArgumentType_SmallFloats to %f" % (None, None, self.value))



class ArgumentType_Float0to100(ArgumentType_Abstract):
    '''
    going to try and use uniform distribution more in the init and in mutate.
    maybe also have a way to do 'fine tune' mutation so it mutates to a more local number.
    also limit values from 0 to 100
    '''
    def __init__(self, value=None):
        if value is None:
            self.mutate_unif100()
        else:
            self.value = value
        ezLogging.debug("%s-%s - Initialize ArgumentType_Float0to100 Class to %f" % (None, None, self.value))


    def mutate_unif100(self):
        self.value = rnd.uniform(0,100)


    def mutate_unif_local(self):
        # make it a range of 10
        low = self.value-5
        high = self.value+5
        self.value = rnd.uniform(low, high)
        # force value to be within 0 to 100
        if (self.value < 0) or (self.value > 100):
            self.mutate_unif100()


    def mutate(self):
        roll = rnd.random()
        if roll < 2/3:
            self.mutate_unif100()
        else:
            self.mutate_unif_local()
        ezLogging.debug("%s-%s - Mutated ArgumentType_Float0to100 to %f" % (None, None, self.value))



class ArgumentType_Int0to100(ArgumentType_Float0to100):
    '''
    same as ArgumentType_Float0to100 but forced as an int
    '''
    def __init__(self, value=None):
        super().__init__(value)
        self.make_int()


    def make_int(self):
        self.value = int(self.value)


    def mutate(self):
        super().mutate()
        self.make_int()



class ArgumentType_Float0to1(ArgumentType_Abstract):
    '''
    like ArgumentType_Float0to100 but go from [0 to 1)
    mutate is just random uniform 0 to 1...may have to introduce fine tuneing...who knows
    '''
    def __init__(self, value=None):
        if value is None:
            self.mutate_unif1()
        else:
            self.value = value
        ezLogging.debug("%s-%s - Initialize ArgumentType_Float0to1 Class to %f" % (None, None, self.value))


    def mutate_unif1(self):
        self.value = np.random.random() #NOTE: [0,1) not (0,1)


    def mutate_unif_local(self):
        low = self.value-.05
        high = self.value+.05
        self.value = rnd.uniform(low, high)
        # force value to be within 0 to 1
        if (self.value < 0) or (self.value > 1):
            self.mutate_unif1()


    def mutate(self):
        roll = rnd.random()
        if roll < 2/3:
            self.mutate_unif1()
        else:
            self.mutate_unif_local()
        ezLogging.debug("%s-%s - Mutated ArgumentType_Float0to1 to %f" % (None, None, self.value))



class ArgumentType_Float0to10(ArgumentType_Float0to1):
    '''
    go from [0 to 10)
    '''
    def __init__(self, value=None):
        if value is None:
            self.mutate_unif1()
        else:
            self.value = value
        ezLogging.debug("%s-%s - Initialize ArgumentType_Float0to10 Class to %f" % (None, None, self.value))


    def mutate_unif10(self):
        self.value = np.random.random()*10 #NOTE: [0,10) not (0,10)


    def mutate_unif_local(self):
        low = self.value-.5
        high = self.value+.5
        self.value = rnd.uniform(low, high)
        # force value to be within 0 to 1
        if (self.value < 0) or (self.value > 1):
            self.mutate_unif1()


    def mutate(self):
        roll = rnd.random()
        if roll < 2/3:
            self.mutate_unif1()
        else:
            self.mutate_unif_local()
        ezLogging.debug("%s-%s - Mutated ArgumentType_Float0to10 to %f" % (None, None, self.value))



class ArgumentType_Int0to25(ArgumentType_Abstract):
    '''
    Augmentor.Pipeline.Rotate has a [0,25] limit so I'm using this to define that range

    NOTE:
        np.random.randint(low, high) ->[low, high)
        np.random.random_integers(low, high) -> [low,high]
    '''
    def __init__(self, value=None):
        if value is None:
            self.mutate_unif_int25()
        else:
            self.value = value
        ezLogging.debug("%s-%s - Initialize ArgumentType_Int0to25 Class to %f" % (None, None, self.value))


    def mutate_unif_int25(self):
        self.value = rnd.random_integers(low=0, high=25)


    def mutate_unif_localint(self):
        # make it a range of 10
        low = self.value-5
        high = self.value+5
        self.value = rnd.random_integers(low, high)
        # force value to be within 0 to 100
        if (self.value < 0) or (self.value > 25):
            self.mutate_unif_int25()


    def mutate(self):
        roll = rnd.random()
        if roll < 2/3:
            self.mutate_unif_int25()
        else:
            self.mutate_unif_localint()
        ezLogging.debug("%s-%s - Mutated ArgumentType_Int0to25 to %f" % (None, None, self.value))

        
        
class ArgumentType_Int1to10(ArgumentType_Abstract):
    '''
    [1,2,3,4,5,6,7,8,9,10]
    NOTE:
        np.random.randint(low, high) ->[low, high)
        np.random.random_integers(low, high) -> [low,high]
    '''
    def __init__(self, value=None):
        if value is None:
            self.mutate_unif_int10()
        else:
            self.value = value
        ezLogging.debug("%s-%s - Initialize ArgumentType_Int1to10 Class to %f" % (None, None, self.value))


    def mutate_unif_int10(self):
        self.value = rnd.random_integers(low=1, high=10)


    def mutate_unif_localint(self):
        # make it a range of 6
        low = self.value-3
        high = self.value+3
        self.value = rnd.random_integers(low, high)
        # force value to be within 0 to 100
        if (self.value < 1) or (self.value > 10):
            self.mutate_unif_int10()


    def mutate(self):
        roll = rnd.random()
        if roll < 2/3:
            self.mutate_unif_int10()
        else:
            self.mutate_unif_localint()
        ezLogging.debug("%s-%s - Mutated ArgumentType_Int1to10 to %f" % (None, None, self.value))


class ArgumentType_Int1to5(ArgumentType_Abstract):
    '''
    [1,2,3,4,5]
    NOTE:
        np.random.randint(low, high) ->[low, high)
        np.random.random_integers(low, high) -> [low,high]
    '''
    def __init__(self, value=None):
        if value is None:
            self.mutate_unif_int5()
        else:
            self.value = value
        ezLogging.debug("%s-%s - Initialize ArgumentType_Int1to5 Class to %f" % (None, None, self.value))


    def mutate_unif_int5(self):
        self.value = rnd.random_integers(low=1, high=5)


    def mutate_unif_localint(self):
        # make it a range of 2
        low = self.value-1
        high = self.value+1
        self.value = rnd.random_integers(low, high)
        # force value to be within 0 to 5
        if (self.value < 1) or (self.value > 5):
            self.mutate_unif_int5()


    def mutate(self):
        roll = rnd.random()
        if roll < 2/3:
            self.mutate_unif_int5()
        else:
            self.mutate_unif_localint()
        ezLogging.debug("%s-%s - Mutated ArgumentType_Int1to5 to %f" % (None, None, self.value))


class ArgumentType_LimitedFloat0to1(ArgumentType_Abstract):
    '''
    limiting ArgumentType_Float0to1 so that our only choices are (0,1] every 0.05
    NOTE that np.random.random() or np.random.uniform() is [0,1)

    This is good for setting nonzero probability
    '''
    def __init__(self, value=None):
        if value is None:
            self.value = None
            self.mutate()
        else:
            self.value = value
        ezLogging.debug("%s-%s - Initialize ArgumentType_LimitedFloat0to1 Class to %f" % (None, None, self.value))


    def mutate(self):
        delta = 0.05
        choices = list(np.arange(0, 1, delta) + delta) #[0.05, 0.1, ..., 0.95, 1.0]
        if self.value in choices:
            choices.remove(self.value) # works in-place
        self.value = np.random.choice(choices)
        ezLogging.debug("%s-%s - Mutated ArgumentType_LimitedFloat0to1 to %f" % (None, None, self.value))

class ArgumentType_PyTorchKernelSize(ArgumentType_Abstract):
    '''
    quick way to pick [1,3,5]
    '''
    def __init__(self, value=None):
        if value is None:
            self.value = None
            self.mutate()
        else:
            self.value = value
        ezLogging.debug("%s-%s - Initialize ArgumentType_PyTorchKernelSize Class to %f" % (None, None, self.value))


    def mutate(self):
        choices = [1,3,5]
        if self.value in choices:
            choices.remove(self.value) # works in-place
        self.value = np.random.choice(choices)
        ezLogging.debug("%s-%s - Mutated ArgumentType_PyTorchKernelSize to %f" % (None, None, self.value))

class ArgumentType_PyTorchPaddingSize(ArgumentType_Abstract):
    '''
    quick way to pick [0, 2, 4, -1], if -1 is chosen, should use automatic padding to cancel out kernel
    '''
    def __init__(self, value=None):
        if value is None:
            self.value = None # This way, we can mutate to None as well
            self.mutate()
        else:
            self.value = value
        ezLogging.debug("%s-%s - Initialize ArgumentType_PyTorchPaddingSize Class to %f" % (None, None, self.value))


    def mutate(self):
        choices = [0, 2, 4, -1]
        if self.value in choices:
            choices.remove(self.value) # works in-place
        self.value = np.random.choice(choices)
        ezLogging.debug("%s-%s - Mutated ArgumentType_PyTorchPaddingSize to %f" % (None, None, self.value))

class ArgumentType_PyTorchActivation(ArgumentType_Abstract):
    '''
    Encodes Pytorch common activation functions

    returns the actual function
    '''
    def __init__(self, value=None):
        if value is None:
            self.value = None
            self.mutate()
        else:
            self.value = value
            self.get_name()
        ezLogging.debug("%s-%s - Initialize ArgumentType_PyTorchActivation Class to %s" % (None, None, self.name))


    def get_name(self):
        if self.value is None:
            self.name = "None"
        else:
            self.name = self.value.__qualname__


    def mutate(self):
        from torch import nn
        choices = [nn.ReLU, nn.LeakyReLU, nn.Tanh, None]
        if self.value in choices:
            choices.remove(self.value) # works in-place
        self.value = np.random.choice(choices)
        self.get_name()
        ezLogging.debug("%s-%s - Mutated ArgumentType_PyTorchActivation to %s" % (None, None, self.name))

class ArgumentType_TrainingStepsShort(ArgumentType_Abstract):
    '''
    Quick way to pick a low number of training steps, used for SimGAN pretraining
    '''
    def __init__(self, value=None):
        if value is None:
            self.value = None # This way, we can mutate to None as well
            self.mutate()
        else:
            self.value = value
        ezLogging.debug("%s-%s - Initialize ArgumentType_TrainingStepsShort Class to %f" % (None, None, self.value))


    def mutate(self):
        # choices = [100, 200, 300, 400, 500]
        # if self.value in choices:
        #     choices.remove(self.value) # works in-place

        choices = [50] # NOTE: for testing, don't leave this

        self.value = np.random.choice(choices)
        ezLogging.debug("%s-%s - Mutated ArgumentType_TrainingStepsShort to %f" % (None, None, self.value))

class ArgumentType_TrainingStepsMedium(ArgumentType_Abstract):
    '''
    Quick way to pick a medium number of training steps, used for SimGAN training
    '''
    def __init__(self, value=None):
        if value is None:
            self.value = None # This way, we can mutate to None as well
            self.mutate()
        else:
            self.value = value
        ezLogging.debug("%s-%s - Initialize ArgumentType_TrainingStepsMedium Class to %f" % (None, None, self.value))


    def mutate(self):
        # choices = [1000, 2000, 3000, 4000, 5000]
        # if self.value in choices:
        #     choices.remove(self.value) # works in-place

        choices = [50] # NOTE: for testing, don't leave this
        self.value = np.random.choice(choices)
        ezLogging.debug("%s-%s - Mutated ArgumentType_TrainingStepsMedium to %f" % (None, None, self.value))

class ArgumentType_LearningRate(ArgumentType_Abstract):
    '''
    quick way to pick a learning rate value, used for simgans
    '''
    def __init__(self, value=None):
        if value is None:
            self.value = None # this way, we can mutate to none as well
            self.mutate()
        else:
            self.value = value
        ezLogging.debug("%s-%s - Mutated ArgumentType_LearningRate to %f" % (None, None, self.value))


    def mutate(self):
        choices = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
        if self.value in choices:
            choices.remove(self.value) # works in-place
        self.value = np.random.choice(choices)
        ezLogging.debug("%s-%s - Mutated ArgumentType_LearningRate to %f" % (None, None, self.value))

class ArgumentType_Placeholder(ArgumentType_Abstract):
    '''
    A placeholder argument type, used for simgan train config operator because we need to have an input
    '''
    def __init__(self, value=None):
        self.value = value

    def mutate(self):
        ezlogging.debug("Called mutate on placeholder")
