'''
root/code/block_definitions/utilities/tools.py

Overview:
Just a miscellaneous file to hold any methods useful to in the block_definitions folder.
'''

### packages
import numpy as np
import logging

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.utilities.custom_logging import ezLogging


def build_weights(method_dict):
    '''
    I wanted to make assigning weights to things as intuitive for the user as possible.
    Ideally the user would just assign weights [0,1) to everything, and make sure that all the
    weights add up to 1 exactly so it's an easy assignment. But say you now want to add a new primitive,
    then you have to go and change the weights for everything.
    Instead here, we give the option to the user to assign a weight of 1, which tells us here
    that we should go and assign the weight for anything that is < 1, then calculate the remainder
    so that the sum is equal to 1, and then equally distribute that remainder to all methods that
    gave a weight of 1.
    
    Here is an example:
    input = {method0: 0.5,
             method1: 1,
             method2: 1,
             method3: 1,
             method4: 1}
    output = {method0: 0.5,
              method1: 0.125,
              method2: 0.125,
              method3: 0.125,
              method4: 0.125}
             
    '''
    ezLogging.debug("%s-%s - Inside build_weights" % (None, None))

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
                ezLogging.error("%s-%s - Current sum of prob/weights for %s is > 1" % (None, None, meth_type))
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
        ezLogging.error("%s-%s - Total sum of prob/weights for %s is < .99" % (None, None, method))
        # TODO, maybe just normalize instead of exit()
        exit()
    else:
        weights[-1] += remainder

    return methods, weights