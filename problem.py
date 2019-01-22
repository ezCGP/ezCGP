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
problem.score_min = 1e-1


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
x_train = [np.float64(1), np.random.uniform(low=0.25, high=2, size=200)]
y_train = goalFunction(x_train[1])

x_test = np.random.uniform(low=0.25, high=2, size=20)
y_test = goalFunction(x_test)

skeleton_block = {
    'nickname': 'regression_block',
    'setup_dict_ftn': {
        operators.add_ff2f: {'prob': 1},
        operators.add_fa2a: {'prob': 1},
        operators.add_aa2a: {'prob': 1},
        operators.sub_ff2f: {'prob': 1},
        operators.sub_fa2a: {'prob': 1},
        operators.sub_aa2a: {'prob': 1},
        operators.mul_ff2f: {'prob': 1},
        operators.mul_fa2a: {'prob': 1},
        operators.mul_aa2a: {'prob': 1}}, # TODO replace this with info from operator_dict?
    'setup_dict_arg': {
        arguments.argInt: {'prob': 1}}, #arg count set to 0 though
    'setup_dict_mate': {
        mate.Mate.dont_mate: {'prob': 1, 'args': []}},
    'setup_dict_mut': {
        mut.Mutate.mutate_singleInput: {'prob': 1, 'args': []},
        #mut.Mutate.mutate_singleArg: {'prob': 1, 'args': []},
        mut.Mutate.mutate_singleFtn: {'prob': 1, 'args': []},},
    'operator_dict': operators.operDict,
    'block_input_dtypes': [np.float64, np.ndarray],
    'block_outputs_dtypes': [np.ndarray],
    'block_main_count': 40,
    'block_arg_count': 2,
    'block_mut_prob': 1,
    'block_mate_prob': 0
}

skeleton_genome = {
    'input': [np.float64, np.ndarray],
    'output': [np.ndarray],
    1: skeleton_block
}
""" block inputs
 nickname,
 setup_dict_ftn, setup_dict_arg, setup_dict_mate, setup_dict_mut,
 operator_dict, block_input_dtypes, block_outputs_dtypes, block_main_count, block_arg_count,
 block_mut_prob, block_mate_prob, 
 tensorblock_flag=False, learning_required=False, num_classes=None
 """