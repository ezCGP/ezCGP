'''
See evolve_only_module.py
This file will be used to parse the config file
and then dynamically produce things needed by the
evolution
'''
### packages
import os
import json
import pdb
import random

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from codes.block_definitions.shapemeta.block_shapemeta import BlockShapeMeta_Abstract
from misc.dynamic_problem_supplements import datatypes_template, primitives_template



class Dynamic():
    def __init__(self, config_filepath):
        if not os.path.exists(config_filepath):
            print("Given config file doesn't seem to exist:\n%s" % config_filepath)
            pdb.set_trace()

        # parse config file
        with open(config_filepath, "r") as f:
            config = json.load(f)
        
        for key, value in config.items():
            '''
            functions
            selection
            mdf_prefix
            parameters
            root_type
            root_primitive
            fitness
            initial_population
            mdf_postfix
            config
            types
            root_methods
            '''
            self.__dict__[key] = value


    def define_datatypes(self):
        '''
        Will use the BlockArguments_Auto() class to populate args,
        so the define_primitives() ftn better produce a valid operator_dict
        to use

        * int
        * float
        * str

        Actually...going to change my approach. Going to think of it more like
        input data types rather than argument/hyperparamter types
        
        say we had a dict of types like:
        {'SomeClass': int,
         'AnotherClass': dict}
        '''
        for dtype_name, dtype in types_dict.items():
            # holy crap that was way easier than i thought
            globals()[dtype_name] = type(dtype_name, (dtype,), {})
        
        # or follow datatypes.py where I write all the custom classes to a new file that can be imported in


    def define_primitives(self):
        '''
        Some existing BlockOperators_Dynamic() class will look for a
        file in this directory; we will write/population that file here.

        Load in some list of template primitives and write from there.
        '''
        pass


    def define_shapemeta(self, name, input_dtypes, output_dtypes, main_count=50):
        '''
        gonna define shapemeta here so we can dynamically change
        input/output datatypes
        
        So this will return an un-instantiated class with type 'name'
        '''
        class BlockShapeMeta_Custom(BlockShapeMeta_Abstract):
            def __init__(self):
                BlockShapeMeta_Abstract.__init__(self,
                                                 input_dtypes,
                                                 output_dtypes,
                                                 main_count)

        return type(name, (BlockShapeMeta_Custom,), {})
