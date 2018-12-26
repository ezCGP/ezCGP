### main.py

# packages:
from collections import defaultdict
from copy import copy, deepcopy
import operator
import random
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import scoop
import pickle

# other python scripts:
from configuration import *
from operators import *
from cgp_generator import start_run
import individual
from problem import Problem
#from evaluation_log import my_logging

# JASON'S CODE
from fromGPFramework.data_v2 import *




# Data Files
train_FILENAME = "data/train_adult.csv.gz"
test_FILENAME = "data/test_adult.csv.gz"
validateTrain_FILENAME = "data/val_train_adult.csv.gz"
validateTest_FILENAME = "data/val_test_adult.csv.gz"

# Read in Data Files
train = load_feature_data_from_file(train_FILENAME) #Returns a GTMOEPData Object
test = load_feature_data_from_file(test_FILENAME)
validate_train = load_feature_data_from_file(validateTrain_FILENAME)
validate_test = load_feature_data_from_file(validateTest_FILENAME)

#global data
train_data = GTMOEPDataPair(train_data=train , test_data=test)
inputs = [train_data]
test_data = GTMOEPDataPair(train_data=validate_train , test_data=validate_test)
test_inputs = [test_data]


if __name__ == '__main__':

    solutions = [] # store the final population from each run
    for run in range(runs):
        
        # i think i need this...from jason's code
        if compute == "scoop":
            scoop._control.execQueue.highwatermark = -1
            scoop._control.execQueue.lowwatermark = -1
        else:
            pass

        solutions.append(start_run(run, seed+run, inputs, test_inputs))
        print("CGP Run #", run, "has converged", flush=True)

        final_file = "outputs/FinalPopulation_run%s" % (run)
        with open(final_file, 'wb') as p:
            for indv in solutions[run]:
                pickle.dump(indv, p)

        # find individual with best overal accuracy (3rd objective)
        min_index = 0
        min_value = referencePoint[2]
        for i, indv in enumerate(solutions[run]):
            if list(indv.fitness.values)[2] < min_value:
                min_index = i
            else:
                pass
        best = deepcopy(solutions[run][i])

        # validate new model with test data
        try:
            best.findActive()
            final_fitness, to_write = individual.testEvaluate(best, test_inputs)
        except:
            print("Unexpected error:", sys.exc_info()[1], flush=True)
            raise
            exit()

        print("\n\nFinal Evaluation and Fintess:\n", to_write, flush=True)

        # lets also plot the pareto fronts!
