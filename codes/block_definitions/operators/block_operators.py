'''
root/code/block_definitions/operators/block_operators.py

Overview:
Here we define the 'scope' of the operators: which primitives is the block allowed to use, with what probability will each be selected, and what is the operator_dictionary for that primitive.
You'll note in the utilities folder that unlike argument_types that has only one file, there are a lot of operator files. This allows us to group primitives together into groups; specifically group by module dependencies and potentially by what different blocks would use. Then note how we don't import all these operators here; rather, each class defines it's own list of operator modules to import, and BlockOperators_Abstract.import_operator_scripts() will import the scripts to global. This way, if I'm solving a symbolic regression problem, I don't have to import tensorflow or keras modules unnecessarily.

Rules:
At the very least will need a list of modules to import and can apply set_equal_weights() to get all the primitives in the modules. Otherwise create the weight dict and pass it into init_from_weight_dict().
'''

### packages
from typing import List
from copy import deepcopy
import importlib
import inspect

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.utilities.custom_logging import ezLogging
from codes.block_definitions.utilities import tools



class BlockOperators_Abstract():
    '''
    Note that the main output from this class will be the operator_dict, a list of all the operators available to the block, and a list of the same size with the 'probability' that the respective operator will be used.
    '''
    def __init__(self):
        ezLogging.debug(
            "%s-%s - Initialize BlockOperators_Abstract Class" % (None, None))
        self.operator_dict = {}
        self.operators = []
        self.weights = []


    def init_from_weight_dict(self, weight_dict):
        '''
        like with BlockArguments_Abstract, we have a method to take in a weight_dict that maybe have values equal to 1, and then tools.build_weights will clean up the weights to proper floats between 0 and 1.
        '''
        ezLogging.debug("%s-%s - Inside init_from_weight_dict" % (None, None))
        operators, weights = tools.build_weights(weight_dict)
        self.operators = operators
        self.weights = weights


    def import_operator_scripts(self, module_names, module_aliases=None):
        '''
        vars() returns a dictionary of local variables...starts out empty inside a function

        vars() -> locals()
        want to eventually change globals()

        note that sys.path is a 'global' variable so the paths added earlier in the file can be used here
        '''
        ezLogging.debug("%s-%s - Inside import_operator_scripts" %
                        (None, None))
        if module_aliases is None:
            module_aliases = deepcopy(module_names)
        else:
            assert(len(module_aliases) == len(module_names)
                   ), "module names and aliases need to be the same length"

        for name, alias in zip(module_names, module_aliases):
            #globals()[alias] = __import__(name)
            # going to use importlib.import_module instead of __import __ because of convention and to do better absolute/relative imports
            globals()[alias] = importlib.import_module(
                "codes.block_definitions.utilities.%s" % name)
            # what about globals().update({alias: ...})
            self.operator_dict.update(globals()[alias].operator_dict)


    def get_all_functions(self, module):
        '''
        this is a good fool proof way of adding a primitive to an operator script in utilities and making sure it gets used by the evolution...if we forget to manually add it to the weight_dict then it won't get used.
        here we assume that all primitives in a module are good for the evolution so we grab them all and return it. It is up to the user to then add weights to each of the primitives, unless this was called through set_equal_weights().
        '''
        ezLogging.debug("%s-%s - Inside get_all_functions" % (None, None))
        all_functions = []
        # returns list of tuples of everything in that module
        for name, execute in inspect.getmembers(globals()[module]):
            if (inspect.isfunction(execute)) and (execute.__module__.endswith(module)) and (execute in self.operator_dict):
                # check if what we are pulling is a function, then make sure it is a function defined in that module
                # as oposed to something imported like dirname from os.path
                all_functions.append(execute)

        ezLogging.debug("%s-%s - Imported %i methods from %s" %
                        (None, None, len(all_functions), module))
        return all_functions


    def set_equal_weights(self, module):
        ezLogging.debug("%s-%s - Inside set_equal_weights" % (None, None))
        weight_dict = {}
        for func in self.get_all_functions(module):
            weight_dict[func] = 1

        return weight_dict



class BlockOperators_SymbRegressionOpsNoArgs(BlockOperators_Abstract):
    '''
    Simple numpy operators that do not require arguments; so we should have a block that takes in the data as one input, and at least one float/int as another input to use.
    '''
    def __init__(self):
        ezLogging.debug(
            "%s-%s - Initialize BlockOperators_SymbRegressionOpsNoArgs Class" % (None, None))
        BlockOperators_Abstract.__init__(self)

        modules = ['operators_symbregression_noargs']
        self.import_operator_scripts(modules)

        weight_dict = {}
        for module in modules:
            weight_dict.update(self.set_equal_weights(module))

        self.init_from_weight_dict(weight_dict)



class BlockOperators_SymbRegressionOpsWithArgs(BlockOperators_Abstract):
    '''
    Basically the same primitives as BlockOperators_SymbRegressionOpsNoArgs but the operator dict calls for arguments; so our block would only have the data as an input.
    '''
    def __init__(self):
        ezLogging.debug(
            "%s-%s - Initialize BlockOperators_SymbRegressionOpsWithArgs Class" % (None, None))
        BlockOperators_Abstract.__init__(self)

        modules = ['operators_symbregression_args']
        self.import_operator_scripts(modules)

        weight_dict = {}
        for module in modules:
            weight_dict.update(self.set_equal_weights(module))

        self.init_from_weight_dict(weight_dict)



class BlockOperators_Gaussian(BlockOperators_Abstract):
    '''
    literally only one operator...just summing gaussians
    '''
    def __init__(self):
        ezLogging.debug(
            "%s-%s - Initialize BlockOperators_Gaussian Class" % (None, None))
        BlockOperators_Abstract.__init__(self)

        modules = ['operators_gaussian_args']
        self.import_operator_scripts(modules)

        weight_dict = {}
        for module in modules:
            weight_dict.update(self.set_equal_weights(module))

        self.init_from_weight_dict(weight_dict)



class BlockOperators_DataAugmentation(BlockOperators_Abstract):
    '''
    augment our data to increase dataset size for training only
    '''
    def __init__(self):
        ezLogging.debug(
            "%s-%s - Initialize BlockOperators_DataAugmentation Class" % (None, None))
        BlockOperators_Abstract.__init__(self)

        modules = ['operators_Augmentor_augmentation']
        self.import_operator_scripts(modules)

        weight_dict = {}
        for module in modules:
            weight_dict.update(self.set_equal_weights(module))

        self.init_from_weight_dict(weight_dict)



class BlockOperators_DataPreprocessing(BlockOperators_Abstract):
    '''
    preprocess images prior to feed to some classifier block
    '''
    def __init__(self):
        ezLogging.debug(
            "%s-%s - Initialize BlockOperators_DataPreprocessing Class" % (None, None))
        BlockOperators_Abstract.__init__(self)

        modules = ['operators_Augmentor_preprocessing']
        self.import_operator_scripts(modules)

        weight_dict = {}
        for module in modules:
            weight_dict.update(self.set_equal_weights(module))

        self.init_from_weight_dict(weight_dict)



class BlockOperators_Augmentor_TransferLearning(BlockOperators_Abstract):
    '''
    pass data through some pretrained network
    '''
    def __init__(self):
        ezLogging.debug(
            "%s-%s - Initialize BlockOperators_TransferLearning Class" % (None, None))
        BlockOperators_Abstract.__init__(self)

        modules = ['operators_Augmentor_TFKeras_transferlearning']
        self.import_operator_scripts(modules)

        weight_dict = {}
        for module in modules:
            weight_dict.update(self.set_equal_weights(module))

        self.init_from_weight_dict(weight_dict)



class BlockOperators_TFKeras_TransferLearning(BlockOperators_Abstract):
    '''
    pass data through some pretrained network
    '''
    def __init__(self):
        ezLogging.debug(
            "%s-%s - Initialize BlockOperators_Keras_TransferLearning Class" % (None, None))
        BlockOperators_Abstract.__init__(self)

        modules = ['operators_TFKeras_transferlearning']
        self.import_operator_scripts(modules)

        weight_dict = {}
        for module in modules:
            weight_dict.update(self.set_equal_weights(module))

        self.init_from_weight_dict(weight_dict)



class BlockOperators_TFKeras_TransferLearning_CIFAR(BlockOperators_Abstract):
    '''
    pass data through some pretrained network
    '''
    def __init__(self):
        ezLogging.debug(
            "%s-%s - Initialize BlockOperators_Keras_TransferLearning Class" % (None, None))
        BlockOperators_Abstract.__init__(self)

        modules = ['operators_TFKeras_transferlearning']
        self.import_operator_scripts(modules)

        weight_dict = {}
        for module in modules:
            weight_dict.update(self.set_equal_weights(module))

        # remove inception() for CIFAR dataset since image size is too small
        weight_dict[operators_TFKeras_transferlearning.inception] = 0

        self.init_from_weight_dict(weight_dict)



class BlockOperators_TFKeras(BlockOperators_Abstract):
    def __init__(self):
        ezLogging.debug(
            "%s-%s - Initialize BlockOperators_TFKeras Class" % (None, None))
        BlockOperators_Abstract.__init__(self)

        modules = ['operators_TensorFlow_Keras']
        self.import_operator_scripts(modules)

        weight_dict = {}
        for module in modules:
            weight_dict.update(self.set_equal_weights(module))

        weight_dict[operators_TensorFlow_Keras.conv2D_layer] = 0.8 #set 80% prob that primitive will be conv2d

        self.init_from_weight_dict(weight_dict)



class BlockOperators_Dense(BlockOperators_Abstract):
    def __init__(self):
        ezLogging.debug(
            "%s-%s - Initialize BlockOperators_Dense Class" % (None, None))
        BlockOperators_Abstract.__init__(self)

        modules = ['operators_TensorFlow_basiclayers']
        self.import_operator_scripts(modules)

        weight_dict = {}
        for module in modules:
            weight_dict.update(self.set_equal_weights(module))

        weight_dict[operators_TensorFlow_basiclayers.dense_layer] = 0.8

        self.init_from_weight_dict(weight_dict)
