'''
root/code/block_definitions/arguments/block_arguments.py

Overview:
In the first ever iteration of creating a CGP framework, it was suggested by Dr. Greg Rohling to remove hyperparamters from the genome and keep them in their own space and perform basic GA on them instead; that way, all the primitives only deal with data manipulation or classification rather than also including basic operations like addition or multiplication to evolve hyperparamter floats or booleans. The problem with this is that it leaves a lot of room for experimentation with how exactly we build out these list of arguments; so a lot more work has to be done to optimize what arguments are introduced and how they can be evolved with GA.

Each block of each individual will have a .args attribute in it's genetic material which will be a random list of arguments/hyperparameters. What this class does, is define the scope of that list: how many arguments, which data types and how much of each. It also specifies the argument data type at each index of the .args list so that every individual will share the same data type but different value at an index.

The way this class works is that the user is expected to provide 2 things:
* arg_count: int to be the length of the .arg list
* arg_dict: dict with keys being the argument data type, and value being the percent share of .args that will be that arg data type
Then init_from_weight_dict() will take those, run it through tools.build_weights to clean up the weight values, and then send to set_arg_types() to fill a list to .arg_types. That list will be the uninstantiated version of .args that all individuals will share to initialize their own .args at their init.

Rules:
Be sure to follow the format to init the abstract class, assign the arg_count and arg dictionary, and then run init_from_weight_dict.
'''

### packages
from typing import List
import inspect
import numpy as np

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.block_definitions.utilities import tools
from codes.block_definitions.utilities import argument_types
from codes.utilities.custom_logging import ezLogging



class BlockArguments_Abstract():
    '''
    Note that this is not an ABC so the user isn't expected to write in their own methods for a new class.
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockArguments_Abstract Class" % (None, None))
        self.arg_count = 0
        self.each_type = []
        self.each_weight = []
        self.arg_types = []


    def init_from_weight_dict(self, weight_dict):
        '''
        This should be the most common method used to initialize this block def.
        It takes in a dictionary of arg data types (key) and the percent (value)
        of the .args that it should populate, and then passes that to set_arg_types
        to fill in .arg_types.
        Note how we first pass it into tools.build_weights. This is because we allow the
        user to assign a weight of '1' in their weight_dict; tools.build_weights goes and
        assigns the 'real' weights of each type after accounting for the '1' weights which
        allow for equal distribution of weight after all non-1 weights have been set.
        '''
        ezLogging.debug("%s-%s - Inside init_from_weight_dict; weight_dict: %s" % (None, None, weight_dict))
        args, weights = tools.build_weights(weight_dict)

        ''' Issue #258
        A while ago, we noticed how random values would sparadically change even though we were seeded.
        It was tracked back to the arg_types changing because we depended on the order returned from
        a for loop over the items of a dictionary which will not gaurentee the same order.
        So now we sort the arg types based of their string representation and that should hold true
        no matter the order of the .items() call
        '''
        str_names = []
        for name in args:
            str_names.append(str(name))
        str_order = np.argsort(str_names)

        self.each_type = np.array(args)[str_order].tolist()
        self.each_weight = np.array(weights)[str_order].tolist()
        self.set_arg_types()


    def get_all_classes(self, module='argument_types'):
        '''
        This is just a helper function to set_equal_weights() to grab all the classes declared in
        the given module. Pretty sexy but will likely never get used.
        '''
        ezLogging.debug("%s-%s - Inside get_all_classes; module: %s" % (None, None, module))
        vals = inspect.getmembers(globals()[module], inspect.isclass)
        # vals will be a list of tuples (name, value)...we want the value
        all_classes = []
        for name, value in vals:
            all_classes.append(value)

        return all_classes


    def set_equal_weights(self, module='argument_types'):
        '''
        if a user was sort of lazy and just wanted to use all possible arguments and
        give them equal weights, then they would just call this function from their
        inherited class' init method.
        this returns a weight_dict, and then the user would pass that to init_from_Weight_dict()
        to complete the init
        '''
        ezLogging.debug("%s-%s - Inside set_all_equal_weights; module: %s" % (None, None, module))
        weight_dict = {}
        for arg_type in self.get_all_classes(module):
            weight_dict[arg_type] = 1

        return weight_dict


    def set_arg_types(self):
        '''
        given a list of unique argument data types and another list giving the percent share of the
        arg_count, this method will fill out a list of .arg_types to be used to initialize the .arg
        of blocks/individuals.
        '''
        ezLogging.debug("%s-%s - Inside set_arg_types" % (None, None))
        start_point = 0
        end_point = 0
        self.arg_types = [None]*self.arg_count
        for arg_class, arg_weight in zip(self.each_type, self.each_weight):
            end_point += int(arg_weight*self.arg_count)
            for arg_index in range(start_point, end_point):
                self.arg_types[arg_index] = arg_class
            start_point = end_point

        if end_point != self.arg_count:
            # prob some rounding errors then
            sorted_byweight = np.argsort(self.each_weight)[::-1] # sort then reverse to go from largest to smallest
            for i, arg_index in enumerate(range(end_point, self.arg_count)):
                arg_class = self.each_type[sorted_byweight[i]]
                self.arg_types[arg_index] = arg_class
        else:
            pass



class BlockArguments_Auto(BlockArguments_Abstract):
    '''
    See if we can get the argument info to be automatically populated by passing in the operator dict.
    Kinda sucks we have to pass in operator_dict which means that the operator_def has to somehow 
    already been initialized or just initialized again for to initialize this class. best way for now I guess
    '''
    def __init__(self, operator_dict, multiplier=5):
        ezLogging.debug("%s-%s - Initialize BlockArguments_Auto Class" % (None, None))
        BlockArguments_Abstract.__init__(self)
        self.multiplier = multiplier

        # get set of all arg types
        arg_types = []
        for method, method_dict in operator_dict.items():
            arg_types += method_dict['args']
        arg_types = list(set(arg_types))
        self.arg_count = multiplier*len(arg_types)

        # set equal weights
        arg_dict = {}
        for arg in arg_types:
            arg_dict[arg] = 1

        # init
        self.init_from_weight_dict(arg_dict)



class BlockArguments_Size50(BlockArguments_Abstract):
    '''
    super simple implementation...50 args: 25 ints, and 25 power of 2s
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockArgumentsSize50 Class" % (None, None))
        BlockArguments_Abstract.__init__(self)
        self.arg_count = 50
        arg_dict = {argument_types.ArgumentType_Ints: 1,
                    argument_types.ArgumentType_Pow2: 1}
        self.init_from_weight_dict(arg_dict)



class BlockArguments_NoArgs(BlockArguments_Abstract):
    '''
    this will be used for when our operators/primitives should never need arguments,
    so we would never populate .args and it will stay an empty list
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockArgumentsNoArgs Class" % (None, None))
        BlockArguments_Abstract.__init__(self)



class BlockArguments_SmallFloatOnly(BlockArguments_Abstract):
    '''
    this will be used for when our operators/primitives should never need arguments,
    so we would never populate .args and it will stay an empty list
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockArgumentsSmallFloatOnly Class" % (None, None))
        BlockArguments_Abstract.__init__(self)
        self.arg_count = 20
        arg_dict = {argument_types.ArgumentType_SmallFloats: 1}
        self.init_from_weight_dict(arg_dict)



class BlockArguments_Gaussian(BlockArguments_Abstract):
    '''
    floats 0-100 for peak location
    ints 0-100 for curve intensity
    floats 0-1 for the std
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockArguments_Gaussian Class" % (None, None))
        BlockArguments_Abstract.__init__(self)
        self.arg_count = 10*3*10 # 10 curves, 3 args each, x10
        arg_dict = {argument_types.ArgumentType_Float0to100: 1,
                    argument_types.ArgumentType_Int0to100: 1,
                    argument_types.ArgumentType_Float0to1: 1}
        self.init_from_weight_dict(arg_dict)



class BlockArguments_DataAugmentation(BlockArguments_Abstract):
    '''
    usage tally:
    argument_types.ArgumentType_LimitedFloat0to1 - lllll lllll lllll ll
    argument_types.ArgumentType_Int1to10 - lll
    argument_types.ArgumentType_Int0to25 - ll
    argument_types.ArgumentType_Float0to10 - lllll l
    argument_types.ArgumentType_Bool - l
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockArguments_DataAugmentation Class" % (None, None))
        BlockArguments_Abstract.__init__(self)
        self.arg_count = 30*3
        arg_dict = {argument_types.ArgumentType_LimitedFloat0to1: 0.5, # 17/30 of all args
                    argument_types.ArgumentType_Int1to10: 1,
                    argument_types.ArgumentType_Int0to25: 1,
                    argument_types.ArgumentType_Float0to10: 0.2, # 6/30
                    argument_types.ArgumentType_Bool: 1}
        self.init_from_weight_dict(arg_dict)



class BlockArguments_DataPreprocessing(BlockArguments_Abstract):
    '''
    usage tally:
    argument_types.ArgumentType_FilterSize - llll
    argument_types.ArgumentType_Bool - l
    argument_types.ArgumentType_Int1to10 - l
    argument_types.ArgumentType_Float0to100- lll
    argument_types.ArgumentType_LimitedFloat0to1 - ll
    argument_types.ArgumentType_Int0to25 - ll
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockArguments_DataPreprocessing Class" % (None, None))
        BlockArguments_Abstract.__init__(self)
        self.arg_count = 13*3
        arg_dict = {argument_types.ArgumentType_FilterSize: 0.33, # 4/13
                    argument_types.ArgumentType_Bool: 1,
                    argument_types.ArgumentType_Int1to10: 1,
                    argument_types.ArgumentType_Float0to100: 0.25, # 3/13
                    argument_types.ArgumentType_LimitedFloat0to1: 0.15, #2/13
                    argument_types.ArgumentType_Int0to25: 1}
        self.init_from_weight_dict(arg_dict)



class BlockArguments_TransferLearning(BlockArguments_Abstract):
    '''
    usage tally:
    argument_types.ArgumentType_Int0to25 - l
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockArguments_TransferLearning Class" % (None, None))
        BlockArguments_Abstract.__init__(self)
        self.arg_count = 1*3
        arg_dict = {argument_types.ArgumentType_Int0to25: 1}
        self.init_from_weight_dict(arg_dict)



class BlockArguments_TFKeras(BlockArguments_Abstract):
    '''
    usage tally:
    argument_types.ArgumentType_Pow2 - llll
    argument_types.ArgumentType_TFFilterSize - llll
    argument_types.ArgumentType_TFActivation - llll
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockArguments_TFKeras Class" % (None, None))
        BlockArguments_Abstract.__init__(self)
        self.arg_count = 12*5
        arg_dict = {argument_types.ArgumentType_Pow2: 1,
                    argument_types.ArgumentType_TFFilterSize: 1,
                    argument_types.ArgumentType_TFActivation: 1,
                    argument_types.ArgumentType_TFPoolSize: 1,
                    argument_types.ArgumentType_Float0to1: 1}
        self.init_from_weight_dict(arg_dict)



class BlockArguments_Dense(BlockArguments_Abstract):
    def __init__(self):
        ezLogging.debug(
            "%s-%s - Initialize BlockArguments_Dense Class" % (None, None))
        BlockArguments_Abstract.__init__(self)
        self.arg_count = 12*4
        arg_dict = {argument_types.ArgumentType_Float0to1: 1,
                    argument_types.ArgumentType_Pow2: 1,
                    argument_types.ArgumentType_TFFilterSize: 1,
                    argument_types.ArgumentType_TFActivation: 1}
        self.init_from_weight_dict(arg_dict)



class BlockArguments_SimGAN_Refiner(BlockArguments_Abstract):
    '''
    usage tally:
    // TODO: keep argument usage tally updated
    argument_types.ArgumentType_Pow2 - ll
    argument_types.ArgumentType_PyTorchFilterSize - l
    argument_types.ArgumentType_PyTorchActivation - l
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockArguments_SimGAN_Refiner Class" % (None, None))
        BlockArguments_Abstract.__init__(self)
        self.arg_count = 60
        arg_dict = {argument_types.ArgumentType_Pow2: 0.25, # Assigns this a 50% chance and then splits the remainin 50% between the last 2
                    argument_types.ArgumentType_PyTorchKernelSize: 1,
                    argument_types.ArgumentType_PyTorchStrideSize: 1,
                    argument_types.ArgumentType_PyTorchPaddingSize: 1,
                    argument_types.ArgumentType_PyTorchActivation: 1,
                    argument_types.ArgumentType_Bool: 1,
                   }
        self.init_from_weight_dict(arg_dict)


class BlockArguments_SimGAN_Discriminator(BlockArguments_Abstract):
    '''
    usage tally:
    // TODO: keep argument usage tally updated
    argument_types.ArgumentType_Pow2 - ll
    argument_types.ArgumentType_PyTorchFilterSize - l
    argument_types.ArgumentType_PyTorchActivation - l
    argument_types.ArgumentType_PyTorchPaddingSize - l
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockArguments_SimGAN_Discriminator Class" % (None, None))
        BlockArguments_Abstract.__init__(self)
        self.arg_count = 70
        arg_dict = {argument_types.ArgumentType_Pow2: 0.3, 
                    argument_types.ArgumentType_PyTorchKernelSize: 0.15,
                    argument_types.ArgumentType_PyTorchStrideSize: 0.15,
                    argument_types.ArgumentType_PyTorchPaddingSize: 0.15,
                    argument_types.ArgumentType_PyTorchActivation: 0.15,
                    argument_types.ArgumentType_Int0to25: 1,
                    argument_types.ArgumentType_LimitedFloat0to1: 1,
                   }
        self.init_from_weight_dict(arg_dict)



class BlockArguments_SimGAN_Train_Config(BlockArguments_Abstract):
    '''
    usage tally:
    argument_types.ArgumentType_TrainingSteps: 1
    argument_types.ArgumentType_PretrainingSteps: 11 
    argument_types.ArgumentType_Int1to5: 11
    argument_types.ArgumentType_LearningRate: 111
    argument_types.ArgumentType_Bool: 1
    '''
    def __init__(self):
        ezLogging.debug("%s-%s - Initialize BlockArguments_SimGAN_Train_Config Class" % (None, None))
        BlockArguments_Abstract.__init__(self)
        self.arg_count = 10*5
        arg_dict = {argument_types.ArgumentType_TrainingSteps: 1.0/10,
                    argument_types.ArgumentType_PretrainingSteps: 2.0/10,
                    argument_types.ArgumentType_Int1to5: 2.0/10,
                    argument_types.ArgumentType_LearningRate: 3.0/10,
                    argument_types.ArgumentType_Bool: 1
                   }
        self.init_from_weight_dict(arg_dict)
