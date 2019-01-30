# problem

import numpy as np
#from scipy.stats import weibull_min
import scipy.stats as scst
from sklearn.metrics import confusion_matrix

import operators
import arguments
import mutate_methods as mut
import mate_methods as mate

# constants
generation_limit = 199
score_min = 1e-1


def goalFunction(x):
    #y = scst.weibull_min(1.79).pdf(x)
    y = 1/x
    return y

def scoreFunction(predict, actual):
    error = actual-predict
    rms_error = np.sqrt(np.mean(np.square(error)))
    max_error = np.max(np.abs(error))
    return rms_error, max_error

# play with difference sizes, and different distribution
x_train = [np.float64(-1), np.random.uniform(low=0.25, high=2, size=200)]
y_train = goalFunction(x_train[1])

x_test = np.random.uniform(low=0.25, high=2, size=20)
y_test = goalFunction(x_test)


# NOTE: a lot of this is hastily developed and I do hope to improve the 'initialization'
#structure of the genome; please note your own ideas and we'll make that a 'project' on github soon

skeleton_block = { #this skeleton defines a SINGLE BLOCK of a genome
    'nickname': 'regression_block',
    'setup_dict_ftn': {
        #declare which primitives are available to the genome,
        #and assign a 'prob' so that you can control how likely a primitive will be used;
        #prob: float btwn 0 and 1 -> assigns that prob to that primitive...the sum can't be more than 1
        #prob: 1 -> equally distribute the remaining probability amoung all those remaining (hard to explain, sorry)
        operators.add_ff2f: {'prob': 1},
        operators.add_fa2a: {'prob': 1},
        operators.add_aa2a: {'prob': 1},
#        operators.sub_ff2f: {'prob': 1},
#        operators.sub_fa2a: {'prob': 1},
#        operators.sub_aa2a: {'prob': 1},
        operators.mul_ff2f: {'prob': 1},
        operators.mul_fa2a: {'prob': 1},
        operators.mul_aa2a: {'prob': 1}}, # TODO replace this with info from operator_dict?
    'setup_dict_arg': {
        #if you have an 'arguments genome', declare which argument-datatypes should fill the argument genome
        #not used for now...arguments genome still needs to be tested
        arguments.argInt: {'prob': 1}}, #arg count set to 0 though
    'setup_dict_mate': {
        #declare which mating methods are available to genomes
        mate.Mate.dont_mate: {'prob': 1, 'args': []}},
    'setup_dict_mut': {
        #declare which mutation methods are available to the genomes
        mut.Mutate.mutate_singleInput: {'prob': 1, 'args': []},
        #mut.Mutate.mutate_singleArg: {'prob': 1, 'args': []},
        mut.Mutate.mutate_singleFtn: {'prob': 1, 'args': []},},
    'operator_dict': operators.operDict, #further defines what datatypes what arguments are required for each primitive
    'block_input_dtypes': [np.float64, np.ndarray], #placeholder datatypes so that the genome can be built off datatypes instead of real data
    'block_outputs_dtypes': [np.ndarray],
    'block_main_count': 40,
    'block_arg_count': 2, #not used...no primitives require arguments
    'block_mut_prob': 1, #mutate genome with probability 1...always
    'block_mate_prob': 0 #mate with probability 0...never
}

skeleton_genome = { # this defines the WHOLE GENOME
    'input': [np.float64, np.ndarray],
    'output': [np.ndarray],
    1: skeleton_block
}
