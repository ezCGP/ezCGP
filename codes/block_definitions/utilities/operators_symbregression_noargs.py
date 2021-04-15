'''
root/codes/block_definitions/utilities/operators...

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

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
    return np.add(a,b)
operator_dict[add_ff2f] = {"inputs": [np.float64, np.float64],
                           "output": np.float64,
                           "args": []
                          }


def add_fa2a(a,b):
    return np.add(a,b)
operator_dict[add_fa2a] = {"inputs": [np.float64, np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


def add_aa2a(a,b):
    return np.add(a,b)
operator_dict[add_aa2a] = {"inputs": [np.ndarray, np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


### Subtraction
def sub_ff2f(a,b):
    return np.subtract(a,b)
operator_dict[sub_ff2f] = {"inputs": [np.float64, np.float64],
                           "output": np.float64,
                           "args": []
                          }


def sub_fa2a(a,b):
    return np.subtract(a,b)
operator_dict[sub_fa2a] = {"inputs": [np.float64, np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


def sub_aa2a(a,b):
    return np.subtract(a,b)
operator_dict[sub_aa2a] = {"inputs": [np.ndarray, np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


### Multiplication
def mul_ff2f(a,b):
    return np.multiply(a,b)
operator_dict[mul_ff2f] = {"inputs": [np.float64, np.float64],
                           "output": np.float64,
                           "args": []
                          }


def mul_fa2a(a,b):
    return np.multiply(a,b)
operator_dict[mul_fa2a] = {"inputs": [np.float64, np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


def mul_aa2a(a,b):
    return np.multiply(a,b)
operator_dict[mul_aa2a] = {"inputs": [np.ndarray, np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


def div_aa2a(a,b):
    return np.divide(a,b)
operator_dict[div_aa2a] = {"inputs": [np.ndarray, np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }


def power_aa2a(a,b):
    return a**np.abs(b)
operator_dict[power_aa2a] = {"inputs": [np.ndarray, np.ndarray],
                           "output": np.ndarray,
                           "args": []
                          }