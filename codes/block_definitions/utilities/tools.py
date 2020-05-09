'''
root/code/block_definitions/utilities/tools.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
import numpy as np

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root


def build_weights(method_dict):
    '''
    TODO MORE

    expecting a dict like this:
     method_dict = { method1: weight1,
                     method2: weight2,}
    '''
    prob_remaining = 1.0
    methods = [None] * len(method_dict)
    weights = [None] * len(method_dict)
    equally_distribute = []
    for i, (meth_type, prob) in enumerate(method_dict.items()):
        methods[i] = meth_type
        if prob <= 0:
            weights[i] = 0
            continue
        elif prob < 1:
            prob_remaining -= prob
            if prob_remaining < 0:
                print("UserInputError: current sum of prob/weights for %s is > 1" % meth_type)
                exit()
            else:
                weights[i] = prob
        else:
            # user wants this prob to be equally distributed with whatever is left
            equally_distribute.append(i)
    # we looped through all methods, now equally distribute the remaining amount
    if len(equally_distribute) > 0:
        eq_weight = round(prob_remaining/len(equally_distribute), 4)
        for i in equally_distribute:
            weights[i] = eq_weight
    # now clean up any rounding errors by appending any remainder to the last method
    remainder = 1 - sum(weights)
    if remainder > .01:
        print("UserInputError: total sum of prob/weights for %s is < .99" % method)
        exit()
    else:
        weights[-1] += remainder

    return methods, weights