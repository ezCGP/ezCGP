### operators.py

# packages:
import numpy as np

# other python scripts:
from fromGPFramework.methods_v3 import knn_scikit, bayes_scikit, omp_scikit, svc_scikit, trees_scikit, random_forest_scikit, best_linear_unbiased_estimate
from fromGPFramework.methods_v3 import kmeans_cluster_scikit, spectral_cluster_scikit, gmm_scikit, my_arg_max, my_arg_min, my_depth_estimate, single_learner
from fromGPFramework.signal_methods_v2 import *
from arguments import * #argInt, argFloat, etc 
from fromGPFramework.data_v2 import GTMOEPDataPair

# dictionary to define data types for all nodes and operators
operDict = {
    "output": [GTMOEPDataPair],
    "input": [GTMOEPDataPair]
}


# NEW way of defining operators

def add_ff2f(a,b):
    return np.add(a,b)
#operDict[add_ff2f] = [[np.float64, np.float64], np.float64]

def add_fa2a(a,b):
    return np.add(a,b)
#operDict[add_fa2a] = [[np.float64,np.ndarray], np.ndarray]
#operDict[add_fa2a] = {"inputs": [np.ndarray],
#						"outputs": np.ndarray,
#						"args": [FloatSmall],
#						"num_args": 3}


def add_aa2a(a,b):
    return np.add(a,b)
#operDict[add_aa2a] = [[np.ndarray,np.ndarray], np.ndarray]
#operDict[add_aa2a] = {"inputs": [np.ndarray, np.ndarray],
#						"outputs": np.ndarray,
#						"args": [],
#						"num_args": 3}

def sub_ff2f(a,b):
    return np.subtract(a,b)
#operDict[sub_ff2f] = [[np.float64, np.float64], np.float64]

def sub_fa2a(a,b):
    return np.subtract(a,b)
#operDict[sub_fa2a] = [[np.float64,np.ndarray], np.ndarray]
#operDict[sub_fa2a] = {"inputs": [np.ndarray],
#						"outputs": np.ndarray,
#						"args": [FloatSmall],
#						"num_args": 3}

def sub_aa2a(a,b):
    return np.subtract(a,b)
#operDict[sub_aa2a] = [[np.ndarray,np.ndarray], np.ndarray]
#operDict[sub_aa2a] = {"inputs": [np.ndarray, np.ndarray],
#						"outputs": np.ndarray,
#						"args": [],
#						"num_args": 3}

def mul_ff2f(a,b):
    return np.multiply(a,b)
#operDict[mul_ff2f] = [[np.float64, np.float64], np.float64]

def mul_fa2a(a,b):
    return np.multiply(a,b)
#operDict[mul_fa2a] = [[np.float64,np.ndarray], np.ndarray]
#operDict[mul_fa2a] = {"inputs": [np.ndarray],
#						"outputs": np.ndarray,
#						"args": [FloatSmall],
#						"num_args": 3}

def mul_aa2a(a,b):
    return np.multiply(a,b)
#operDict[mul_aa2a] = [[np.ndarray,np.ndarray], np.ndarray]
#operDict[mul_aa2a] = {"inputs": [np.ndarray, np.ndarray],
#						"outputs": np.ndarray,
#						"args": [],
#						"num_args": 3}


#############################
### add the primitives from signal_methods
# but go to gp_framework_helper_v2 to get the info
# and then define the data type classes in arguements.py
# for the float, int, and bool args, maybe change them to 'myInt' or 'argInt' and create a new class like that
# I think that 'num_args' is not needed here anymore and is moved to where the arguement class is being defineed

operDict[my_if_then_else] = {"inputs": [GTMOEPDataPair, GTMOEPDataPair, GTMOEPDataPair],
							"args": [argInt],
							"outputs": GTMOEPDataPair,
							"name": 'myIfThenElse'}

operDict[my_averager] = {"inputs": [GTMOEPDataPair],
							"args": [argInt, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myAverager'}

operDict[my_diff] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myDiff'}

operDict[my_auto_corr] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myDiff'}

operDict[my_dct] = {"inputs": [GTMOEPDataPair],
							"args": [argInt, argInt, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myDCT'}

operDict[window_hann] = {"inputs": [GTMOEPDataPair],
							"args": [argBool, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myHann'}

operDict[window_hamming] = {"inputs": [GTMOEPDataPair],
							"args": [argBool, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myHamming'}

operDict[window_tukey] = {"inputs": [GTMOEPDataPair],
							"args": [argFloat, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myTukey'}

operDict[window_cosine] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myCosine'}

operDict[window_lanczos] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myLanczos'}

operDict[window_triangular] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myTriangular'}


operDict[window_bartlett] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myBartlett'}

operDict[window_gaussian] = {"inputs": [GTMOEPDataPair],
							"args": [argFloat, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myGaussian'}

operDict[window_bartlett_hann] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myBartlettHann'}

operDict[window_blackman] = {"inputs": [GTMOEPDataPair],
							"args": [argFloat, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myBlackman'}

operDict[window_kaiser] = {"inputs": [GTMOEPDataPair],
							"args": [argFloat, argBool, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myKaiser'}

operDict[window_planck_taper] = {"inputs": [GTMOEPDataPair],
							"args": [argFloat, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myPlanckTaper'}

operDict[window_nuttall] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myNuttall'}

operDict[window_blackman_harris] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myBlackmanHarris'}

operDict[window_blackman_nuttall] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myBlackmanNuttall'}

operDict[window_flat_top] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myFlatTop'}

#operDict[sub_sample_data] = {"inputs": [GTMOEPDataPair],
#							"args": [argInt, argInt, TriState],
#							"outputs": GTMOEPDataPair,
#							"name": 'mySubSampleData',
#							"num_args": 3}

operDict[my_fft] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myFFT'}

operDict[my_norm] = {"inputs": [GTMOEPDataPair],
							"args": [argInt, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myNorm'}

#operDict[my_rms_2d] = {"inputs": [GTMOEPDataPair],
#							"args": [argInt, TriState],
#							"outputs": GTMOEPDataPair,
#							"name": 'myRMS2D',
#							"num_args": 3}

operDict[my_sum] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'mySum'}

operDict[my_cum_sum] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myCumSum'}

operDict[my_prod] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myProd'}

operDict[my_cum_prod] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myCumProd'}

operDict[my_abs] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myAbs'}

operDict[my_log] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myLog'}

operDict[my_arcsine] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myArcSineMath'}

operDict[my_arctangent] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myArcTangentMath'}

operDict[my_sine] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'mySineMath'}

operDict[my_cosine] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myCosineMath'}

operDict[my_tangent] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myTangentMath'}

operDict[my_exp] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myExpMath'}

operDict[my_round] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myRound'}

operDict[my_cross_corr] = {"inputs": [GTMOEPDataPair, GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myCrossCorr'}

operDict[my_dwt] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myDWT'}

operDict[remove_feature] = {"inputs": [GTMOEPDataPair],
							"args": [argInt],
							"outputs": GTMOEPDataPair,
							"name": 'myRemoveFeature'}

operDict[my_kalman_filter] = {"inputs": [GTMOEPDataPair],
							"args": [argFloat, argFloat, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'kalman'}

operDict[my_linear_predictive_coding] = {"inputs": [GTMOEPDataPair],
							"args": [argInt, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'LPC'}

operDict[my_wiener_filter] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'wiener'}

operDict[my_savitzky_golay_filter] = {"inputs": [GTMOEPDataPair],
							"args": [argInt, argInt, argInt, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'savGol'}

operDict[my_pca] = {"inputs": [GTMOEPDataPair],
							"args": [argInt, argBool],
							"outputs": GTMOEPDataPair,
							"name": 'myPCA'}

operDict[my_peak_finder] = {"inputs": [GTMOEPDataPair],
							"args": [argFloat, argBool, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myPeakFinder'}

operDict[my_informed_search] = {"inputs": [GTMOEPDataPair, GTMOEPDataPair],
							"args": [argInt, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myInformedSearch'}

operDict[cut_data_lead] = {"inputs": [GTMOEPDataPair],
							"args": [argInt, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myCutDataLead'}

operDict[my_rebase] = {"inputs": [GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myRebase'}

# COULDN'T GET HMMLEARN TO INSTALL
#operDict[gaussian_hmm] = {"inputs": [GTMOEPDataPair],
#							"args": [argInt],
#							"outputs": GTMOEPDataPair,
#							"name": 'myGaussianHmm'}

operDict[my_ecdf] = {"inputs": [GTMOEPDataPair],
							"args": [argInt, TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myECDF'}

operDict[select_range] = {"inputs": [GTMOEPDataPair],
							"args": [argInt, argInt],
							"outputs": GTMOEPDataPair,
							"name": 'selectRange'}

operDict[my_concatenate] = {"inputs": [GTMOEPDataPair, GTMOEPDataPair],
							"args": [TriState],
							"outputs": GTMOEPDataPair,
							"name": 'myConcatenate'}

#############################
### add the primitives for machine learning...single_learner
# so single_learner() is defined in methods_v3.py but is added to primitive set in gp_framework
# and LearnerType() is defined in gp_frameowkr which should be stored in args

operDict[single_learner] = {"inputs": [GTMOEPDataPair],
							"args": [LearnerType],
							"outputs": GTMOEPDataPair,
							"name": 'SingleLearner'}

# WILL HAVE TO CHANGE THIS MAYBE...DO WE WANT AS AN OPPORATOR??
#operDict[modifyLearner] = {"inputs": [LearnerType, argInt, argInt],
#							"outputs": LearnerType,
#							"name": 'ModifyLearner'}

#operDict[modifyLearner] = {"inputs": [LearnerType, bool],
#							"outputs": LearnerType,
#							"name": 'ModifyLearner'}

#operDict[modifyLearner] = {"inputs": [LearnerType, float],
#							"outputs": LearnerType,
#							"name": 'ModifyLearner'}
