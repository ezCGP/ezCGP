# machine learners

# first check all fronts and find best indiv.
# then evaluate validation data on the individual

# packages
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import confusion_matrix # type I and II error:
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import time
import gc
#%matplotlib inline

import individual
from fromGPFramework.data_v2 import *
import fromGPFramework.signal_methods_v2 as sm
#with open(FOLDERNAME + files[-1], 'rb') as p:
#    front = pickle.load(p)


FOLDERNAME = "outputs/WedMultiRun0/"
REFERENCE_POINT = [1.,1.]

damaged_files = ["PFront_run0_gen49_vol78.txt"]


def get_front(file):
    front =[]
    #print(FOLDERNAME + file)
    with open(FOLDERNAME + file, 'rb') as g:
        while True:
            try:
                front.append(pickle.load(g))
            except EOFError:
                break
    return front



files = [x for x in os.listdir(FOLDERNAME) if x.startswith('PFront')]

# get generation #s
c=list(files[0].split("_"))
vols = []
gens = []
want_files = []

# find largest volume.
vol1 = 0
vol2 = 0
for i, file in enumerate(files):
    vol = [int(gen[3:].split(".")[0]) for gen in list(file.split("_")) if gen[:3]=="vol"]

    if vol[0] > vol1:
        vol2 = vol1
        vol1 = vol[0]
    elif vol[0] == vol1:
        pass
    elif vol[0] > vol2:
        vol2 = vol[0]
    else:
        pass
# now should have best and second best volume sizes
# go back through and find the files with that size
fronts = []
summary = np.empty((0,5), float)
f = -1
for file in files:
    vol = [int(gen[3:].split(".")[0]) for gen in list(file.split("_")) if gen[:3]=="vol"]

    if vol not in [vol1, vol2] and file not in damaged_files: # <- that file must be damaged
        f += 1
        front = get_front(file)
        fronts.append(front) # keep the full individuals here
        for i, indv in enumerate(front):
            summary = np.append(summary, np.array([[f, i, indv.fitness.values[0], indv.fitness.values[1], indv.fitness.values[2]]]), axis=0)
    else:
        pass
# sort all lists

minIndex = summary[:,-1].argmin()
f, i, xmin, ymin, zmin = summary[minIndex,:]

best = fronts[int(f)][int(i)]

print("\nbest results from training:", best.fitness.values)

################
## load data
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
##################################

try:
    best.findActive()
    final_fitness, to_write = individual.testEvaluate(best, test_inputs)
except:
    print("Unexpected error:", sys.exc_info()[1], flush=True)
    raise
    exit()
print("\n\nFinal Evaluation and Fintess:\n", to_write, flush=True)
# now we have the best estimate from the model


######################################################################################################
##### MACHINE LEARNERS

#### now do random forest
def random_forest_scikit(data_pair, n_estimators=100, class_weight=None):
    """
    Implements a random forest using scikit
    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    #target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    #print('random_forest_scikit: training', training_data.shape);sys.stdout.flush()# print_mem(); sys.stdout.flush()

    forest = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight)
    forest.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    #print('random_forest_scikit: testing', testing_data.shape);sys.stdout.flush()# print_mem(); sys.stdout.flush()
    predicted_classes = forest.predict(testing_data)
    [inst.set_target([target]) for inst, target in
        zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = forest.predict(training_data)
    [inst.set_target([target]) for inst, target in
        zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="Forest")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
        zip(data_pair.get_train_data().get_instances(), target_values)]
    return data_pair




####### SVM
kernels = ['linear','poly','rbf','sigmoid']
#kernel = kernels[learner.learnerParams['kernel']%len(kernels)]
def svc_scikit(data_pair, kernel='rbf'):
    """
    Implements a C-Support vector classification using scikit
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # Check for multiple target values
    #target_value_check(target_values)
    # For debugging purposes let's print out name of the function and dimensions of the data
    #print('svc_scikit: training', training_data.shape);sys.stdout.flush()# print_mem(); sys.stdout.flush()
    #print("going")
    svc = SVC(kernel=kernel)
    #print("done")
    # Putting this in a try due to a number of classes equal to 1 error... this should not be happening
    try:
        svc.fit(training_data, target_values)
    except ValueError as e:
        print(training_data.shape, target_values, np.unique(target_values), target_values.shape); sys.stdout.flush
        raise e
    #print("done2")
    testing_data = data_pair.get_test_data().get_numpy()
    #print('svc_scikit: testing', testing_data.shape);sys.stdout.flush()# print_mem(); sys.stdout.flush()
    predicted_classes = svc.predict(testing_data)
    [inst.set_target([target]) for inst, target in
        zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = svc.predict(training_data)
    [inst.set_target([target]) for inst, target in
        zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="SVC")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
        zip(data_pair.get_train_data().get_instances(), target_values)]
    return data_pair

def knn_scikit(data_pair, k=3, weights='uniform'):
    """
    Returns the result of a kNN machine learner using scikit
    http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])
    # For debugging purposes let's print out name of the function and dimensions of the data
    #print('knn_scikit: training', training_data.shape); sys.stdout.flush()#print_mem(); sys.stdout.flush()
    neigh = KNeighborsClassifier(n_neighbors=k, weights=weights)
    neigh.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    #print('knn_scikit: testing', testing_data.shape);sys.stdout.flush()# print_mem(); sys.stdout.flush()
    predicted_classes = neigh.predict(testing_data)
    [inst.set_target([target]) for inst, target in
        zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = neigh.predict(training_data)
    [inst.set_target([target]) for inst, target in
        zip(data_pair.get_train_data().get_instances(), trained_classes)]
    data_pair = sm.makeFeatureFromClass(data_pair, name="kNN")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
        zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair

def bayes_scikit(data_pair):
    """
    Returns the result of a naive bayes machine learner using scikit
    http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes
    """
    data_pair = copy.deepcopy(data_pair)
    training_data = data_pair.get_train_data().get_numpy()
    target_values = np.array([inst.get_target()[0] for
                              inst in data_pair.get_train_data().get_instances()])

    # For debugging purposes let's print out name of the function and dimensions of the data
    #print('bayes_scikit: training', training_data.shape);sys.stdout.flush()# print_mem(); sys.stdout.flush()
    bayes = GaussianNB()
    bayes.fit(training_data, target_values)
    testing_data = data_pair.get_test_data().get_numpy()
    #print('bayes_scikit: testing', testing_data.shape);sys.stdout.flush()# print_mem(); sys.stdout.flush()
    predicted_classes = bayes.predict(testing_data)
    [inst.set_target([target]) for inst, target in
        zip(data_pair.get_test_data().get_instances(), predicted_classes)]

    # Let's make the predictions a feature through use of the make feature from class,
    # But then restore the training data to the class
    # Set the self-predictions of the training data
    trained_classes = bayes.predict(training_data)
    [inst.set_target([target]) for inst, target in
        zip(data_pair.get_train_data().get_instances(), trained_classes)]

    data_pair = sm.makeFeatureFromClass(data_pair, name="bayes")
    # Restore the training data
    [inst.set_target([target]) for inst, target in
        zip(data_pair.get_train_data().get_instances(), target_values)]

    return data_pair


def getFitness(output, inputt):
    actual = np.array([instance._target[0] for instance in inputt[0]._test_data._instances])
    # try to fix predicted so it is ONLY 0 and 1
    predicted = []
    for instance in output._test_data._instances: #self.evaluated[numMain]._test_data._instances:
        val = np.rint([instance._target[0]])
        if val >= 1.:
            predicted.append(1.)
        else:
            predicted.append(0.)
    predicted = np.array(predicted)
    conMat = confusion_matrix(actual, predicted)
    fitness = () 

    # typeII
    type2error = conMat[1,0] / (conMat[0,0] + conMat[1,0])
    fitness += (type2error,) #objective ftn 2

    # typeI ... false positive
    type1error = conMat[0,1] /  (conMat[1,1] + conMat[0,1])
    fitness += (type1error,) #objective ftn 1

    # overal error
    overerror = (conMat[0,1] + conMat[1,0]) / (conMat[0,0] + conMat[1,0] + conMat[0,1] + conMat[1,1])
    fitness += (2*overerror,)

    return fitness



######################################################################################################


class_weights = [None, 'balanced', 'balanced_subsample']
class_weight = class_weights[2%len(class_weights)]
random_forest_output = random_forest_scikit(test_inputs[0])#, 80, class_weight)
RF_test = getFitness(random_forest_output, test_inputs)
# should have the random forest output now
print("fitness for random forest VALIDATION:", RF_test)

time.sleep(15)
gc.collect()

##### NOW DO THE TRAIN DATA
random_forest_output = random_forest_scikit(inputs[0])#, 80, class_weight)
RF_train = getFitness(random_forest_output, inputs)
# should have the random forest output now
print("fitness for random forest TRAIN DATA:", RF_train)

time.sleep(15)
gc.collect()

# SVC
#kernel = kernels[learner.learnerParams['kernel']%len(kernels)]
svc_output = svc_scikit(inputs[0], kernel='rbf')
svc_train = getFitness(svc_output, inputs)
print("fitness for SVC TRAIN DATA:", svc_train)

time.sleep(15)
gc.collect()

# knn
knn_output = knn_scikit(inputs[0], k=3, weights='uniform')
knn_train = getFitness(knn_output, inputs)
print("fitness for knn TRAIN DATA:", knn_train)

time.sleep(15)
gc.collect()

# bayes
bayes_output = bayes_scikit(inputs[0])
bayes_train = getFitness(bayes_output, inputs)
print("fitness for bayes TRAIN DATA:", bayes_train)






exit()


