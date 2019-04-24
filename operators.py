# operators

import numpy as np

# dictionary to define data types for all nodes and operators
operDict = {
    "output": [np.ndarray, np.float64],
    "input": [np.ndarray]
}
#TODO this seems redundant; gotta clean this up

def add_ff2f(a,b):
    return np.add(a,b)
#operDict[add_ff2f] = [[np.float64, np.float64], np.float64]
operDict[add_ff2f] = {"inputs": [np.float64, np.float64],
						"outputs": np.float64,
						"args": []
						}

def add_fa2a(a,b):
    return np.add(a,b)
#operDict[add_fa2a] = [[np.float64,np.ndarray], np.ndarray]
operDict[add_fa2a] = {"inputs": [np.ndarray],
						"outputs": np.ndarray,
						"args": ["argFloat"]
						}
"""
operDict[add_fa2a] = {"inputs": [np.ndarray],
						"outputs": np.ndarray,
						"args": [FloatSmall],
						"num_args": 3}
"""

def add_aa2a(a,b):
    return np.add(a,b)
#operDict[add_aa2a] = [[np.ndarray,np.ndarray], np.ndarray]
operDict[add_aa2a] = {"inputs": [np.ndarray, np.ndarray],
						"outputs": np.ndarray,
						"args": []
						}
#operDict[add_aa2a] = {"inputs": [np.ndarray, np.ndarray],
#						"outputs": np.ndarray,
#						"args": [],
#						"num_args": 3}

def sub_ff2f(a,b):
    return np.subtract(a,b)
#operDict[sub_ff2f] = [[np.float64, np.float64], np.float64]
operDict[sub_ff2f] = {"inputs": [np.float64, np.float64],
						"outputs": np.float64,
						"args": []
						}

def sub_fa2a(a,b):
    return np.subtract(a,b)
#operDict[sub_fa2a] = [[np.float64,np.ndarray], np.ndarray]
operDict[sub_fa2a] = {"inputs": [np.ndarray],
						"outputs": np.ndarray,
						"args": ["argFloat"]
						}
#operDict[sub_fa2a] = {"inputs": [np.ndarray],
#						"outputs": np.ndarray,
#						"args": [FloatSmall],
#						"num_args": 3}

def sub_aa2a(a,b):
    return np.subtract(a,b)
#operDict[sub_aa2a] = [[np.ndarray,np.ndarray], np.ndarray]
operDict[sub_aa2a] = {"inputs": [np.ndarray, np.ndarray],
						"outputs": np.ndarray,
						"args": []
						}
#operDict[sub_aa2a] = {"inputs": [np.ndarray, np.ndarray],
#						"outputs": np.ndarray,
#						"args": [],
#						"num_args": 3}

def mul_ff2f(a,b):
    return np.multiply(a,b)
#operDict[mul_ff2f] = [[np.float64, np.float64], np.float64]
operDict[mul_ff2f] = {"inputs": [np.float64, np.float64],
						"outputs": np.float64,
						"args": []
						}

def mul_fa2a(a,b):
    return np.multiply(a,b)
#operDict[mul_fa2a] = [[np.float64,np.ndarray], np.ndarray]
operDict[mul_fa2a] = {"inputs": [np.ndarray],
						"outputs": np.ndarray,
						"args": ["argFloat"]
						}
#operDict[mul_fa2a] = {"inputs": [np.ndarray],
#						"outputs": np.ndarray,
#						"args": [FloatSmall],
#						"num_args": 3}

def mul_aa2a(a,b):
    return np.multiply(a,b)
#operDict[mul_aa2a] = [[np.ndarray,np.ndarray], np.ndarray]
operDict[mul_aa2a] = {"inputs": [np.ndarray, np.ndarray],
						"outputs": np.ndarray,
						"args": []
						}
#operDict[mul_aa2a] = {"inputs": [np.ndarray, np.ndarray],
#						"outputs": np.ndarray,
#						"args": [],
#						"num_args": 3}