'''
root/problems/problem_tensorflow_keras.py

Overview:
Using a lot of tf.keras version 2+, we will be trying to create a 'baseline' vanilla way of
putting together a 'Problem()' instance.

A lot of this has evolved from:
* https://github.com/ezCGP/ezExperimental/blob/2020S-student-edits/problem_tensorflow.py
* https://github.com/ezCGP/ezExperimental/blob/2020S-student-edits/problem_augmentation.py

Rules:
None
'''
### packages
import numpy as np
import glob
# Fitness imports
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score as accuracy

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems.problem_definition import ProblemDefinition_Abstract
from codes.factory import FactoryDefinition
from data.data_tools.loader import ezDataLoader_CIFAR10
from codes.utilities.custom_logging import ezLogging
from post_process import save_things
# Block Defs
from codes.block_definitions.shapemeta.block_shapemeta import (BlockShapeMeta_DataAugmentation,
                                                               BlockShapeMeta_DataPreprocessing,
                                                               BlockShapeMeta_TFKeras_TransferLearning,
                                                               BlockShapeMeta_TFKeras)
from codes.block_definitions.operators.block_operators import (BlockOperators_DataAugmentation,
                                                               BlockOperators_DataPreprocessing,
                                                               BlockOperators_TFKeras_TransferLearning_CIFAR,
                                                               BlockOperators_TFKeras)
from codes.block_definitions.arguments.block_arguments import (BlockArguments_DataAugmentation,
                                                               BlockArguments_DataPreprocessing,
                                                               BlockArguments_TransferLearning,
                                                               BlockArguments_TFKeras)
from codes.block_definitions.evaluate.block_evaluate import (BlockEvaluate_MiddleBlock,
                                                             BlockEvaluate_MiddleBlock_SkipValidating)
from codes.block_definitions.evaluate.block_evaluate_graph import (BlockEvaluate_TFKeras_TransferLearning,
                                                                   BlockEvaluate_TFKeras_CloseAnOpenGraph)
from codes.block_definitions.mutate.block_mutate import BlockMutate_OptB_4Blocks
from codes.block_definitions.mate.block_mate import BlockMate_WholeOnly_4Blocks, BlockMate_NoMate
# Individual Defs
from codes.individual_definitions.individual_mutate import IndividualMutate_RollOnEachBlock
from codes.individual_definitions.individual_mate import IndividualMate_RollOnEachBlock
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_wAugmentorPipeline_wTensorFlow_OpenCloseGraph



class Problem(ProblemDefinition_Abstract):
    '''
    Vanilla/basic usage of Augmentor pipeline + tf.keras for CIFAR10
    
    Basic block flow:
        1) Data Augmentation only for training (increase dataset size)
        2) Data Preprocessing
        3) Transfer Learning
        4) Custom Keras NN
    '''
    def __init__(self):
        import tensorflow as tf
        # assert(len(tf.config.experimental.list_physical_devices('GPU'))>=1), "GPU NOT FOUND - ezCGP EXITING"

        population_size = 4#20
        number_universe = 1
        factory = FactoryDefinition
        factory_instance = factory()
        mpi = False
        genome_seeds = []
        #genome_seeds = glob.glob("outputs/problem_tfkeras_transferlearning/%s/univ0000/gen_%04d_*.pkl" % ("20201127-145527-8th_run", 1))
        super().__init__(population_size, number_universe, factory, mpi, genome_seeds)
        
        augmentation_block_def = self.construct_block_def(nickname="augmentation_block",
                                                          shape_def=BlockShapeMeta_DataAugmentation,
                                                          operator_def=BlockOperators_DataAugmentation,
                                                          argument_def=BlockArguments_DataAugmentation,
                                                          evaluate_def=BlockEvaluate_MiddleBlock_SkipValidating,
                                                          mutate_def=BlockMutate_OptB_4Blocks,
                                                          mate_def=BlockMate_WholeOnly_4Blocks)

        preprocessing_block_def = self.construct_block_def(nickname="preprocessing_block",
                                                           shape_def=BlockShapeMeta_DataPreprocessing,
                                                           operator_def=BlockOperators_DataPreprocessing,
                                                           argument_def=BlockArguments_DataPreprocessing,
                                                           evaluate_def=BlockEvaluate_MiddleBlock,
                                                           mutate_def=BlockMutate_OptB_4Blocks,
                                                           mate_def=BlockMate_WholeOnly_4Blocks)

        transferlearning_block_def = self.construct_block_def(nickname="transferlearning_block",
                                                           shape_def=BlockShapeMeta_TFKeras_TransferLearning,
                                                           operator_def=BlockOperators_TFKeras_TransferLearning_CIFAR,
                                                           argument_def=BlockArguments_TransferLearning,
                                                           evaluate_def=BlockEvaluate_TFKeras_TransferLearning,
                                                           mutate_def=BlockMutate_OptB_4Blocks,
                                                           mate_def=BlockMate_WholeOnly_4Blocks)

        tensorflow_block_def = self.construct_block_def(nickname="tensorflow_block",
                                                        shape_def=BlockShapeMeta_TFKeras,
                                                        operator_def=BlockOperators_TFKeras,
                                                        argument_def=BlockArguments_TFKeras,
                                                        evaluate_def=BlockEvaluate_TFKeras_CloseAnOpenGraph,
                                                        mutate_def=BlockMutate_OptB_4Blocks,
                                                        mate_def=BlockMate_WholeOnly_4Blocks)
        
        self.construct_individual_def(block_defs=[augmentation_block_def,
                                                  preprocessing_block_def,
                                                  transferlearning_block_def,
                                                  tensorflow_block_def],
                                      mutate_def=IndividualMutate_RollOnEachBlock,
                                      mate_def=IndividualMate_RollOnEachBlock,
                                      evaluate_def=IndividualEvaluate_wAugmentorPipeline_wTensorFlow_OpenCloseGraph)

        self.construct_dataset()


    def construct_dataset(self):
        loader = ezDataLoader_CIFAR10(0.6, 0.2, 0.2)
        self.training_datalist, self.validating_datalist, self.testing_datalist = loader.load()

        
    def objective_functions(self, indiv):
        '''
        :param indiv: individual which contains references to output of training
        :return: None
        
        2 objectives:
            1) accuracy score
            2) f1 score
        
        Old Code:
            dataset = self.data
            _, actual = dataset.preprocess_test_data()
            actual = np.argmax(actual, axis = 1)
            predict = indiv.output
            predict = np.argmax(predict, axis = 1)
            acc_score = accuracy(actual, predict)
            f1 = f1_score(actual, predict, average = "macro")
            indiv.fitness.values = (-acc_score, -f1)  # want to minimize this
        
        With updated code, we expect the last block to return the validation metrics assigned to the Model object,
        so we just need to connect those to the individual's fitness values
        '''
        if indiv.dead:
            indiv.fitness.values = (0,0,0)
        else:
            indiv.fitness.values = tuple(indiv.output)


    def check_convergence(self, universe):
        """
        :param universe:
        :return:
        """
        GENERATION_LIMIT = 2 #50
        SCORE_MIN = 1 - 1e-10

        # only going to look at the 2nd objective value which is f1
        min_firstobjective_index = universe.pop_fitness_scores[:,1].argmin()
        min_firstobjective = universe.pop_fitness_scores[min_firstobjective_index,:]
        ezLogging.warning("Checking Convergence - generation %i, best score: %s" % (universe.generation, min_firstobjective))

        if universe.generation >= GENERATION_LIMIT:
            ezLogging.warning("TERMINATING...reached generation limit.")
            universe.converged = True
        if np.abs(min_firstobjective[0]) > SCORE_MIN:
            ezLogging.warning("TERMINATING...reached minimum scores.")
            universe.converged = True


    def postprocess_generation(self, universe):
        '''
        after each generation, we want to save the scores (plot performance over time)
        and save the population for seeding
        '''
        ezLogging.info("Post Processing Generation Run - saving")
        save_things.save_fitness_scores(universe)
        save_things.save_population(universe)


    def postprocess_universe(self, universe):
        '''
        NOTE that this is not an abstractmethod because the user may choose not to do anything here

        the idea here is that the universe.run() is about to exit but before it does,
        we can export or plot things wrt the final population
        '''
        ezLogging.info("Post Processing Universe Run - pass")
        pass
