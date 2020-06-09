'''
root/codes/block_definitions/utilities/operators...

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
from codes.block_definitions.utilities import argument_types



### init dict
operator_dict = {}


### the only method
def one_gauss(x, peak, std, intensity, ybump=0.0):
	'''
	https://en.wikipedia.org/wiki/Normal_distribution
	expect x to be a numpy array of locations
	peak -> float between 0 and 100
	intensity -> int between 0 and 100
	std -> float between 0 and 1
	'''
	factor = 1/(std*np.sqrt(2*np.pi))
	inside_exp = -0.5 * np.square((x-peak)/std)
	return intensity*factor*np.exp(inside_exp) + ybump
operator_dict[one_gauss] = {"inputs": [np.ndarray],
                            "output": np.ndarray,
                            "args": [argument_types.ArgumentType_Float0to100,
                            		 argument_types.ArgumentType_Float0to1,
                            		 argument_types.ArgumentType_Int0to100]
                           }