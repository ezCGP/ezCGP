'''
Special Problem class for case we want to use evolve_only_module.py
'''
### packages
import os

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
from misc.dynamic_problem_supplements import dynamic_constructor

from codes.block_definitions.operators.block_operators import BlockOperators_Dynamic
from codes.block_definitions.arguments.block_arguments import BlockArguments_Auto
from codes.block_definitions.evaluate.block_evaluate_pytorch import BlockEvaluate_WritePy
from codes.block_definitions.mutate.block_mutate import BlockMutate_OptB
from codes.block_definitions.mate.block_mate import BlockMate_WholeOnly
from codes.individual_definitions.individual_mutate import IndividualMutate_RollOnEachBlock_LimitedMutants
from codes.individual_definitions.individual_mate import IndividualMate_RollOnEachBlock
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_WritePy



class Problem():
    '''
    Not gonna init from ABC because our requirements are different,
    but I will copypaste:
        * parent_selection()
        * construct_block_def()
        * construct_individual_def()
    '''
    def __init__(self, config_filepath):
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
        self.process_config(config_filepath)

        # set input datatypes
        self.construct_dataset()


    def process_config(self, config_filepath):
        '''
        * define custom arg/datatypes
        * define primtives using those types
        * define block_defs
        '''
        dyanmic = dynamic_constructor.Dynamic(config_filepath)
        dynamic.define_argtypes()
        dynamic.define_primitives()
        MyShapeMeta = dynamic.define_shapemeta()

        block_defs = []
        for ith_block, function in enumerate(dynamic.functions):
            block_def = self.construct_block_def(nickname="%s-%i" % (function, ith_block),
                                                 shape_def=MyShapeMeta,
                                                 operator_def=BlockOperators_Dynamic,
                                                 argument_def=BlockArguments_Auto(BlockOperators_Dynamic(), 4),
                                                 evaluate_def=BlockEvaluate_WritePy,
                                                 mutate_def=BlockMutate_OptB(prob_mutate=0.2, num_mutants=2),
                                                 mate_def=BlockMate_WholeOnly(prob_mate=1/len(dynamic.functions)))
            block_defs.append(block_def)


        self.construct_individual_def(block_defs=block_defs,
                                      mutate_def=IndividualMutate_RollOnEachBlock_LimitedMutants,
                                      mate_def=IndividualMate_RollOnEachBlock,
                                      evaluate_def=IndividualEvaluate_WritePy)




    def construct_dataset(self):
        # in our case, don't need data for evaluation
        # but will still define as to not break code
        self.training_datalist = [None]
        self.validating_datalist = [None]
        self.testing_datalist = [None]


    def parent_selection(*args, **kwargs):
        ProblemDefinition_Abstract.parent_selection(*args, **kwargs)


    def construct_block_def(*args, **kwargs):
        ProblemDefinition_Abstract.construct_block_def(*args, **kwargs)


    def construct_individual_def(*args, **kwargs):
        ProblemDefinition_Abstract.construct_individual_def(*args, **kwargs)