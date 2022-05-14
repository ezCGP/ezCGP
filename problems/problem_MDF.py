'''
Special Problem class for case we want to use evolve_only_module.py
'''
### packages
import os
import json

### sys relative to root dir
import sys
from os.path import dirname, realpath
ezCGP_rootdir = dirname(dirname(realpath(__file__)))
#dynamic_supplements = os.path.join(ezCGP_rootdir, "misc", "dynamic_problem_supplements")
sys.path.append(ezCGP_rootdir)
#sys.path.append(dynamic_supplements)


### absolute imports wrt root
from codes.utilities.custom_logging import ezLogging
from problems.problem_definition import ProblemDefinition_Abstract, welless_check_decorator
from codes.factory import FactoryDefinition
#from misc.dynamic_problem_supplements import dynamic_constructor

from codes.block_definitions.shapemeta.block_shapemeta import BlockShapeMeta_Abstract
from codes.block_definitions.operators.block_operators import BlockOperators_MDF
from codes.block_definitions.arguments.block_arguments import BlockArguments_Auto, BlockArguments_NoArgs
from codes.block_definitions.evaluate.block_evaluate import BlockEvaluate_WriteFtn
from codes.block_definitions.mutate.block_mutate import BlockMutate_OptA
from codes.block_definitions.mate.block_mate import BlockMate_WholeOnly
from codes.individual_definitions.individual_mutate import IndividualMutate_RollOnEachBlock_LimitedMutants
from codes.individual_definitions.individual_mate import IndividualMate_RollOnEachBlock
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_WriteMDF



class Problem():
    '''
    Not gonna init from ABC because our requirements are different,
    but I will copypaste:
        * parent_selection()
        * construct_block_def()
        * construct_individual_def()
    '''
    def __init__(self, config_filepath, output_directory):
        # load in standard metadata stuff
        self.pop_size = 4
        self.hall_of_fame_flag = False
        self.number_universe = 1
        self.Factory = FactoryDefinition
        self.mpi = False
        self.genome_seeds = []
        self.maximize_objectives = [True] # <- nonsense/not-needed
        self.num_objectives = 1 # <- nonsense/not-needed

        # load in config stuff
        self.process_config(config_filepath, output_directory=output_directory)

        # set input datatypes
        self.construct_dataset()


    def process_config(self, config_filepath, **kwargs):
        '''
        * define custom arg/datatypes
        * define primtives using those types
        * define block_defs
        '''
        self.config = Config(config_filepath)

        # manually add in any metadata we want to pass in with config dict
        for key, value in kwargs.items():
            self.config.__dict__[key] = value

        # TODO: change format later to actually use config
        self.config.functions = ['temp_fake_function']

        block_defs = []
        for ith_block, function in enumerate(self.config.functions):
            # TODO define required datatypes
            #types_dict = self.config['SOMETHING']
            types_dict = {"MyInt_0": int,
                          "MyInt_1": int}
            input_dtypes = []
            for dtype_name, dtype in types_dict.items():
                globals()[dtype_name] = type(dtype_name, (dtype,), {})
                input_dtypes.append(globals()[dtype_name])

            output_dtypes = [int] #[globals()['MyInt_0']]

            MyShapeMeta = self.define_shapemeta("BlockShapeDynamic_%s" % function,
                                                input_dtypes,
                                                output_dtypes,
                                                main_count=50)

            block_def = self.construct_block_def(nickname="%s-%i" % (function, ith_block),
                                                 shape_def=MyShapeMeta,
                                                 operator_def=BlockOperators_MDF,
                                                 argument_def=BlockArguments_NoArgs(),
                                                 #argument_def=BlockArguments_Auto(BlockOperators_MDF(), 4),
                                                 evaluate_def=BlockEvaluate_WriteFtn,
                                                 mutate_def=BlockMutate_OptA(prob_mutate=0.2, num_mutants=2),
                                                 mate_def=BlockMate_WholeOnly(prob_mate=1/len(self.config.functions)))
            block_defs.append(block_def)

        self.construct_individual_def(block_defs=block_defs,
                                      mutate_def=IndividualMutate_RollOnEachBlock_LimitedMutants,
                                      mate_def=IndividualMate_RollOnEachBlock,
                                      evaluate_def=IndividualEvaluate_WriteMDF)





    def construct_dataset(self):
        # in our case, don't need data for evaluation
        # but will still define as to not break code
        self.training_datalist = [self.config] # TODO !!!! at the individual.evaluate() level, will customize training_datalist for each block
        self.validating_datalist = [None]
        self.testing_datalist = [None]


    def population_selection(*args, **kwargs):
        return ProblemDefinition_Abstract.population_selection(*args, **kwargs)


    def parent_selection(*args, **kwargs):
        return ProblemDefinition_Abstract.parent_selection(*args, **kwargs)


    def construct_block_def(*args, **kwargs):
        return ProblemDefinition_Abstract.construct_block_def(*args, **kwargs)


    def construct_individual_def(*args, **kwargs):
        ProblemDefinition_Abstract.construct_individual_def(*args, **kwargs)


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


class Config():
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
