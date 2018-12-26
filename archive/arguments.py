### arguments.py

# packages
import numpy as np
from numpy import random as r
import random
from copy import copy

# other python scripts
from configuration import perc_args, streamData


### Mutation Methods
'''
    Define a set of mutation methods to be called on to mutate all/any of the argument classes.

    Consider limiting them to be strictly positive or non-negative
'''
#def mut_uniform(self, low=0, high=1, sd=None):
#    if sd != None:
#        low = self.value - sd
#        high = self.value + sd
#    else:
#        pass
#    self.value = r.uniform(low,high)

def mut_uniform(value):
    if value == 0:
        return r.uniform(0,5)
    else:
        low = value*.85
        high = value * 1.15
        return r.uniform(low,high)

def mut_normal(value):
    if value == 0:
        return r.normal(3, 3*.1)
    else:
        mean = value
        sd = value * .1
        return r.normal(mean, sd)




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
arguments = []


class argBool(object):

    def __init__(self):
        self.value = True
        self.num_samples = 10
        self.mutate()

    def mutate(self):
        self.value = random.choice([True, False])
        # or should i force it to pick whatever it isn't...true-->false, false-->true?
arguments.append(argBool)


class argFloat(object):

    def __init__(self):
        roll = r.random_integers(0,2)
        if roll == 0:
            self.value = 5.0
        elif roll == 1:
            self.value = 50.0
        elif roll == 2:
            self.value = 100.0
        self.num_samples = 10
        self.mutate()

    def mutate(self):
        roll = r.random_integers(0,2)
        if roll == 0:
            self.value = mut_normal(self.value)
        elif roll == 1:
            self.value = mut_uniform(self.value)
        else:
            pass
        if self.value < 0:
            self.value *= -1
        else:
            pass
arguments.append(argFloat)


class argInt(object):

    def __init__(self):
        roll = r.random_integers(0,2)
        if roll == 0:
            self.value = 5
        elif roll == 1:
            self.value = 50
        elif roll == 2:
            self.value = 100
        self.num_samples = 10
        self.mutate()

    def mutate(self):
        roll = r.random_integers(0,2)
        if roll == 0:
            self.value = mut_normal(self.value)
        elif roll == 1:
            self.value = mut_uniform(self.value)
        else:
            pass
        if self.value < 1:
            self.value = 1
        else:
            pass
        self.value = int(self.value)
arguments.append(argInt)


# overwrite class LearnerType() and class TriState from gp_framework_helper_v2
from fromGPFramework.gp_framework_helper_v2 import modifyLearner, learnerGen

class LearnerType:
    '''
    This is the argument fed into the primitive operator singler_learner().

    This was originally built for Jason's GP and was then adapted here to work for Rodd's
    CGP in that it must return all relevant attributes under self.value and have a mutate
    method.
    In an effort to maintain as much of Jason's code, self.value returns the self object
    itself. This is unusual syntax and method for passing information but the most simple
    way to unite the two.

    So self.value returns the object itself so it preserves the way that single_learner()
    calls on the attributes after it is read in. So the object returned from self.value
    is the exact same object (same hash) as self. If self.learnerName get's changed,
    it gets changed in self (obviously), and it gets changed in self.value...and for that
    matter, it also gets changed in self.value.value....value.value 
    Not an elegant solution but clever imo
    '''
    def __init__(self):
        learnerName, learnerParams = learnerGen()
        self.learnerName = learnerName
        self.learnerParams = learnerParams
        self.num_samples = 50
        self.value = self #<--RODD ADDED THIS

    def __repr__(self):
        return 'learnerType(\'' + str(self.learnerName) + '\', ' + str(self.learnerParams) + ')'

    def mutate(self):
        if self.learnerParams != None:
            # randomly decide how many and then which ones of the parameters to mutate
            paramLen = len(self.learnerParams)
            mutateCount = random.randint(1, paramLen)
            params = np.random.choice(a=range(paramLen), size=mutateCount, replace=False)
            for pos in params:

                key = list(self.learnerParams.keys())[pos]
                value = self.learnerParams[key]
                if value == "uniform":
                    pos = 0
                    key = list(self.learnerParams.keys())[pos]
                    value = self.learnerParams[key]
                else:
                    pass

                i = random.randint(0,1)

                if i == 0:
                    newValue = mut_uniform(value)
                elif i == 1:
                    newValue = mut_normal(value)
                else:
                    pass

                if newValue<1 and type(value)==int:
                    newValue = 1
                elif type(value) == int:
                    newValue = int(newValue)
                elif newValue < 0:
                    newValue *= -1
                else:
                    pass

                self = modifyLearner(self, newValue=newValue, pos=pos) #defined in gp_framework_helper_v2
        else:
            pass
arguments.append(LearnerType)


class TriState(int):
    '''
    Class which can take on three possible values:
        feature to feature
        stream to stream
        feature to stream

    stream data is used for time dependent data

    if there is no stream data, then TriState needs to be fixed to
    0 (feature to feature) ... set streamData=False in configuration.py
    '''
    def __init__(self):
        self.value = 0
        self.mutate()
        if streamData:
            self.num_samples = 9
        else:
            self.num_samples = 1
    def mutate(self):
        if streamData:
            self.value = random.randint(0,2)
        else: #feature data only
            self.value = 0
arguments.append(TriState)


def build_arg_skeleton(set_numArgs):
    '''
    Each population is built off of the same randomly constructed
    'argument skeleton'...this is a list where each element is a randomly
    defined object of one of the classes listed in 'arguments'.
    Everyone in the population shares the same order of data types, but
    with different values which makes mating of arguments very simple.

    So there are two major options. Either the user defines the size/
    length of the argument list with 'set_numArgs' in configurations.py
    or the argument list is filled with objects in 'arguments' and with
    the count defined by the attribute 'num_samples' (number of samples
    to create for that argument object)

    And if streamData=False, then we don't need multiple objects with 
    class TriState...only 1.

    Also perc_args allows the user to bias the argument skeleton to have
    more LearnerType objects. Since there are mutliple machine learning
    methods tucked under a single object type, it make sense to allow 
    a bias to make sure the skeleton is filled with a well diversed set
    of machine learner types.
    '''
    arguments_skeleton = []
    global perc_args

    if set_numArgs == 0 or type(set_numArgs).__name__ != 'int':
        # then the arg size needs to be calculated by summing all 'num_args' attributes
        numArgs = 0
        temp_arguments = copy(arguments)
        if streamData==False:
            arguments_skeleton.append(TriState())
            arguments_skeleton.append(TriState())
            numArgs = 2
            temp_arguments.remove(TriState)
        else:
            pass

        for arg_type in arguments:
            arg_sample = arg_type()
            arguments_skeleton.append(arg_sample)
            numArgs += arg_sample.num_samples 
            for _ in range(1,arg_sample.num_samples):
                arguments_skeleton.append(arg_type())
        
    else: # then set_numArgs is an int > 0
        numArgs = set_numArgs
        if numArgs * (1-perc_args) - 9 <= (len(arguments)-2) * 2: #see below for why...
            print("You picked too small of a number for 'set_numArgs", flush=True)
            print("set_numArgs:", set_numArgs, flush=True)
            print("number of arguments in list:", len(arguments), flush=True)
            print("We recommend going at least x3 the number above", flush=True)
            exit()
        else:
            pass 
        
        temp_arguments = copy(arguments)

        # fill w/ LearnerType
        i = int(numArgs * perc_args)
        for _ in range(i):
            arguments_skeleton.append(LearnerType())
        temp_arguments.remove(LearnerType)

        # fill w/ TriState
        if streamData:
            arg_sample = TriState()
            arguments_skeleton.append(arg_sample)
            i += arg_sample.num_samples 
            for _ in range(1,arg_sample.num_samples):
                arguments_skeleton.append(TriState())
        else: #just feature data
            arguments_skeleton.append(TriState())
            arguments_skeleton.append(TriState())
            i += 2
        temp_arguments.remove(TriState)

        # at a minimum, numArgs will be 2 times the number of argument types in the list, arguments
        # first fill at 2 arguments samples for each arg_type
        for arg in temp_arguments:
            i+=2   
            arguments_skeleton.append(arg())
            arguments_skeleton.append(arg())

        for _ in range(i,numArgs):
                arg = random.choice(temp_arguments)
                arguments_skeleton.append(arg())

    random.shuffle(arguments_skeleton)
    return arguments_skeleton, numArgs