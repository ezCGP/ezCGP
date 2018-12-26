### problem.py

# packages:
import random
import numpy as np
from copy import deepcopy
from scipy.stats import weibull_min
from sklearn.metrics import confusion_matrix # type I and II error:
import math

# other python scripts:
from configuration import *
from fromGPFramework.data_v2 import *
from evaluation_log import my_logging



# define 'Problem' class
class Problem(object):
    '''
    Define the problem we want to model and the training data.

    This answers:
    * what are we modeling
    * how do we measure convergence...what is being optimized
    
    NOTE: this will be inherited into the 'Individual' class.
    '''


    def __init__(self):

        #self.inputs = [train_data]
        #self.test_inputs = [test_data]
        pass

        # so we want to read in the data and give it trainX as the entire train data
        # and trainY as only the trainY for the training portion of the train data

        # and for testing... we want to give it both train and test to be the respective datasets
        # and then give testY to be only the Y of the test data...if that makes any sense


    def getFitness(self, global_inputs, evaluated):
        '''
        In this step, we want to take the evaluated Output Nodes and apply some metric to
        measure performance of the model.
        
        Let's make an assumption that all objective functions need to be minimized.
        
        What we want to return is a fitness value which we'll pass as a tuple containing an error
        or value for each objective function...remember, all objective functions are assumed to
        be a minimization problem.
        '''
        
        fitness = () #empty tuple
        #if hasattr(self, 'evaluated'):
        #if True:
        # in this case, need to take out the actual categorical data
        if self.has_learned and self.dead==False: #make sure there was at least 1 learner obj and isn't dead

            # NEW / FIXED
            # want to get 'target' data from 'train data' of the output datapair
            actual = np.array([instance._target[0] for instance in global_inputs[0]._test_data._instances])
            # try to fix predicted so it is ONLY 0 and 1
            predicted = []
            for instance in evaluated._test_data._instances: #self.evaluated[numMain]._test_data._instances:
                val = np.rint([instance._target[0]])
                if val >= 1.:
                    predicted.append(1.)
                else:
                    predicted.append(0.)
            predicted = np.array(predicted)


            print(type(actual), len(actual), actual, flush=True)
            print(type(predicted), len(predicted), predicted, flush=True)

            conMat = confusion_matrix(actual, predicted) #dimensions: (actual classification) x (predicted classification) ... (row) x (col)
            print("matrix", conMat, flush=True)
            #wait = input("press enter")

            obsCount = len(actual)

            # type II ... false negative
            type2error = conMat[1,0] / (conMat[0,0] + conMat[1,0])
            fitness += (type2error,) #objective ftn 2

            # typeI ... false positive
            type1error = conMat[0,1] /  (conMat[1,1] + conMat[0,1])
            fitness += (type1error,) #objective ftn 1

            # overal error
            overerror = (conMat[0,1] + conMat[1,0]) / (conMat[0,0] + conMat[1,0] + conMat[0,1] + conMat[1,1])
            fitness += (2*overerror,)

            ## clean up fitness
            temp = list(fitness)
            for fval, fit in enumerate(temp):
                if math.isnan(fit):
                    temp[fval] = referencePoint[fval]
                else:
                    pass
            fitness = tuple(temp)

            print(fitness, flush=True)

        else: #has not learned or is dead
            fitness = tuple(referencePoint) #assign worst possible fitness for minimization
        #else:
        #    pass

        #print("wtf", fitness)
        self.fitness.values = fitness
        self.to_write = my_logging("fitness", [self.to_write, self.fitness.values])