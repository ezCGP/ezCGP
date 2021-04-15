'''
root/codes/block_definitions/utilities/operators...

Overview:
Operators for basic symbolic regression benchmarking as defined
in "A comparitive study on crossover in cartesian genetic programming"

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
import numpy as np

'''
### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
'''


### init dict
operator_dict = {}


### Addition
def add_ff2f(a,b):
    return a+b
operator_dict[add_ff2f] = {"inputs": [np.float64, np.float64],
                           "output": np.float64,
                           "args": []
                          }


def add_fa2a(a,b):
    return a+b
operator_dict[add_fa2a] = {"inputs": [np.float64, np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


def add_aa2a(a,b):
    return a+b
operator_dict[add_aa2a] = {"inputs": [np.ndarray, np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


### Subtraction
def sub_ff2f(a,b):
    return a-b
operator_dict[sub_ff2f] = {"inputs": [np.float64, np.float64],
                           "output": np.float64,
                           "args": []
                          }


def sub_fa2a(a,b):
    return a-b
operator_dict[sub_fa2a] = {"inputs": [np.float64, np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


def sub_aa2a(a,b):
    return a-b
operator_dict[sub_aa2a] = {"inputs": [np.ndarray, np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


### Multiplication
def mul_ff2f(a,b):
    return a*b
operator_dict[mul_ff2f] = {"inputs": [np.float64, np.float64],
                           "output": np.float64,
                           "args": []
                          }


def mul_fa2a(a,b):
    return a*b
operator_dict[mul_fa2a] = {"inputs": [np.float64, np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


def mul_aa2a(a,b):
    return a*b
operator_dict[mul_aa2a] = {"inputs": [np.ndarray, np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


### Division
def div_ff2f(a,b):
    return a/b
operator_dict[div_ff2f] = {"inputs": [np.float64, np.float64],
                           "output": np.float64,
                           "args": []
                          }


def div_fa2a(a,b):
    return a/b
operator_dict[div_fa2a] = {"inputs": [np.float64, np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


def div_aa2a(a,b):
    return a/b
operator_dict[div_aa2a] = {"inputs": [np.ndarray, np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


### sin
def sin_f2f(a):
    # assume radians
    return np.sin(a)
operator_dict[sin_f2f] = {"inputs": [np.float64],
                           "output": np.float64,
                           "args": []
                          }


def sin_a2a(a):
    # assume radians
    return np.sin(a)
operator_dict[sin_a2a] = {"inputs": [np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


### cos
def cos_f2f(a):
    # assume radians
    return np.cos(a)
operator_dict[cos_f2f] = {"inputs": [np.float64],
                           "output": np.float64,
                           "args": []
                          }


def cos_a2a(a):
    # assume radians
    return np.cos(a)
operator_dict[cos_a2a] = {"inputs": [np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


### natural log
def ln_f2f(a):
    # np.log is base e
    return np.log(np.abs(a))
operator_dict[ln_f2f] = {"inputs": [np.float64],
                           "output": np.float64,
                           "args": []
                          }


def ln_a2a(a):
    # np.log is base e
    return np.log(np.abs(a))
operator_dict[ln_a2a] = {"inputs": [np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


### e^x
def exp_f2f(a):
    return np.exp(a)
operator_dict[exp_f2f] = {"inputs": [np.float64],
                           "output": np.float64,
                           "args": []
                          }


def exp_a2a(a):
    return np.exp(a)
operator_dict[exp_a2a] = {"inputs": [np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }
