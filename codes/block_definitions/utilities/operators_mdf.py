'''
mdf operators are a little weird...
Here we are writing individual lines to a function in a python script.
The actual operators only write out a string representation of what we want to happen,
and the inputs are actually just string variable names.
The operator dict 'inputs' and 'args' keys will match the datatype we expect the ftn
to use in the python script that will eventually get written, not the datatypes we get
at evaluate()
'''

### packages


### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.block_definitions.utilities import argument_types
#from codes.utilities.custom_logging import ezLogging

### init dict
operator_dict = {}


def my_add(input_0, input_1, output, *args):
    line = "%s = %s + %s" % (output, input_0, input_1)
    return line

operator_dict[my_add] = {'inputs': [int, int],
                         'args': [],
                         'output': int}


def my_subtract(input_0, input_1, output, *args):
    line = "%s = %s - %s" % (output, input_0, input_1)
    return line

operator_dict[my_subtract] = {'inputs': [int, int],
                              'args': [],
                              'output': int}