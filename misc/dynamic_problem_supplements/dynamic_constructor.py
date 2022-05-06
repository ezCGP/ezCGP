'''
See evolve_only_module.py
This file will be used to parse the config file
and then dynamically produce things needed by the
evolution
'''
### packages
import os
import pdb
import random

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from codes.block_definitions.shapemeta.block_shapemeta import BlockShapeMeta_Abstract
from codes.misc.dynamic_problem_supplements import datatypes_template, primitives_template



class Dynamic():
    def __init__(self, config_filepath):
        if not os.path.exists(config_filepath):
            print("Given config file doesn't seem to exist:\n%s" % config_filepath)
            pdb.set_trace()

        # parse config file
        # TODOOOOO


    def define_argtypes(self):
        '''
        Will use the BlockArguments_Auto() class to populate args,
        so the define_primitives() ftn better produce a valid operator_dict
        to use

        * int
        * float
        * str

        ...how to mutate each?
        '''
        TODO


    def define_primitives(self):
        '''
        Some existing BlockOperators_Dynamic() class will look for a
        file in this directory; we will write/population that file here.

        Load in some list of template primitives and write from there.
        '''
        TODO


    def define_shapemeta(self):
        '''
        gonna define shapemeta here so we can dynamically change
        input/output datatypes
        '''
        class BlockShapeMeta_Custom(BlockShapeMeta_Abstract):
            def __init__(self):
                input_dtypes = [] # TODO
                output_dtypes = [] # TODO
                main_count = 50
                BlockShapeMeta_Abstract.__init__(self,
                                                 input_dtypes,
                                                 output_dtypes,
                                                 main_count)

        return BlockShapeMeta_Custom