'''
codes/utilities/decorators.py
'''

### packages
import time

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from codes.utilities.custom_logging import ezLogging



def stopwatch_decorator(func):
    '''
    decorator to help with logging how long it takes to finish a method
    '''
    def inner(*args, **kwargs):
       	start_clock = time.time()

        output = func(*args, **kwargs)

        end_clock = time.time()
        ezLogging.info("Stopwatch - %s took %.2f seconds" % (func.__name__, end_clock-start_clock))

        return output

    return inner