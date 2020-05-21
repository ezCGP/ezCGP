'''
root/code/block_definitions/block_arguments.py

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

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from codes.block_definitions.utilities import tools
from codes.block_definitions.utilities import argument_types



class BlockArguments_Abstract():
    '''
    Note that this is not an ABC so the user isn't expected to write in their own methods for a new class.
    '''
    def __init__(self):
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
        args, weights = tools.build_weights(weight_dict)
        self.each_type = args
        self.each_weight = weights
        self.set_arg_types()


    def get_all_classes(self, module='argument_types'):
        '''
        This is just a helper function to set_equal_weights() to grab all the classes declared in
        the given module. Pretty sexy but will likely never get used.
        '''
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
        start_point = 0
        end_point = 0
        self.arg_types = [None]*self.arg_count
        for arg_class, arg_weight in zip(self.each_type, self.each_weight):
            end_point += int(arg_weight*self.arg_count)
            for arg_index in range(start_point, end_point):
                self.arg_types[arg_index] = arg_type
            start_point = end_point

        if end_point != self.arg_count:
            # prob some rounding errors then
            sorted_byweight = np.argsort(self.each_weight)[::-1] # sort then reverse to go from largest to smallest
            for i, arg_index in enumerate(range(end_point, self.arg_count)):
                arg_class = self.each_type[sorted_byweight[i]]
                self.arg_types[arg_indx] = arg_class
        else:
            pass



class BlockArgumentsSize50(BlockArguments_Abstract):
    '''
    super simple implementation...50 args: 25 ints, and 25 power of 2s
    '''
    def __init__(self):
        BlockArguments_Abstract.__init__(self)
        self.arg_count = 50
        arg_dict = {argument_types.ArgumentType_Ints: 1,
                    argument_types.ArgumentType_Pow2: 1}
        self.init_from_weight_dict(arg_dict)



class BlockArgumentsNoArgs(BlockArguments_Abstract):
    '''
    this will be used for when our operators/primitives should never need arguments,
    so we would never populate .args and it will stay an empty list
    '''
    def __init__(self):
        BlockArguments_Abstract.__init__(self)