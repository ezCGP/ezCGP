'''
root/problem/problem_definition.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
import os
import glob
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from copy import deepcopy

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from codes.factory import FactoryDefinition
#from codes.universe import UniverseDefinition # can't import because will create import loop
from codes.utilities.custom_logging import ezLogging
from codes.utilities import selections
from codes.genetic_material import IndividualMaterial#, BlockMaterial
from codes.individual_definitions.individual_definition import IndividualDefinition
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_Abstract
from codes.individual_definitions.individual_mutate import IndividualMutate_Abstract
from codes.individual_definitions.individual_mate import IndividualMate_Abstract
from codes.block_definitions.block_definition import BlockDefinition
from codes.block_definitions.evaluate.block_evaluate import BlockEvaluate_Abstract
from codes.block_definitions.mutate.block_mutate import BlockMutate_Abstract
from codes.block_definitions.mate.block_mate import BlockMate_Abstract
from codes.block_definitions.shapemeta.block_shapemeta import BlockShapeMeta_Abstract
from codes.block_definitions.operators.block_operators import BlockOperators_Abstract
from codes.block_definitions.arguments.block_arguments import BlockArguments_Abstract


def welless_check_decorator(func):
    '''
    just a convenient decorator to make it easier for the user to remember to check for dead
    individuals and to quickly assign worst possible scores

    inputs:
        self -> Problem instance
        population -> Population instance
    '''
    def inner(self, population):
        for indiv in population.population:
            if indiv.dead:
                indiv.set_worst_score()
            else:
                but_really_is_dead = False
                # double check just to be sure
                for block in indiv.blocks:
                    if block.dead:
                        indiv.dead = True
                        indiv.set_worst_score()
        
        func(self, population)

    return inner


class ProblemDefinition_Abstract(ABC):
    '''
     * data: training + validation
     * objective ftn(s)
     * define convergence
     * individual definition
     * which universe
    '''
    # Note to user: all abstract methods are not defined 
    # by default, please implement according to preferences
    def __init__(self,
                 population_size: int,
                 number_universe: int,
                 factory_def: FactoryDefinition,
                 mpi: bool=False,
                 genome_seeds: List=[],
                 hall_of_fame_flag: bool=True):
        '''
        genome_seeds:
        * each element in outer list is an inividual to be seeded
        * each element in the inner list is the seed for that positional block of the individual
        * an inner element of 'None' will get a randomly built block
        * an element could be a pkl file instead of another list if we are loading in a whole pickled IndividualMaterial
        
        self.construct_dataset()

        the build out each block and build individual_def
        block_def = self.construct_block()
        self.construct_individual([block_def])
        '''
        self.pop_size = population_size
        self.hall_of_fame_flag = hall_of_fame_flag
        self.number_universe = number_universe
        self.Factory = factory_def
        self.mpi = mpi
        self.genome_seeds = genome_seeds
        self.set_optimization_goals()
        self.number_of_objectives = len(self.maximize_objectives)
        self.construct_dataset()


    @abstractmethod
    def construct_dataset(self):
        '''
        training data + labels
        validating data + labels
        '''
        pass


    @abstractmethod
    def set_optimization_goals(self):
        '''
        Fill in the maximize_objectives attribute as a tuple/list of boolean values
        where if the ith element is True, that indicates that the ith objective
        is to be maximized in the optimization, and False indicates minimization.

        Ex:
        self.maximize_objectives = [True, True, False]

        This tuple will be sued to set the weights and establish the number of
        objectives for the IndividualMaterial.Fitness class,
        And will be used to set the worst possible score for in individual_evaluate
        if the individual is dead.
        '''
        pass



    @abstractmethod
    def objective_functions(self, indiv: IndividualMaterial):
        '''
        save fitness for each individual to IndividualMaterial.fitness.values as tuple
        
        try:
            acc_score = accuracy_score(actual, predict)
            avg_f1_score = f1_score(actual, predict, average='macro')
            return 1 - acc_score, 1 - avg_f1_score
        except ValueError:
            print('Malformed predictions passed in. Setting worst fitness')
            return 1, 1  # 0 acc_score and avg f1_score b/c we want this indiv ignored
        '''
        pass


    @abstractmethod
    def check_convergence(self, universe):
        '''
        whether some threshold for fitness or some max generation count

        set universe.converged to boolean T/F ...True will end the universe run
        '''
        pass


    def population_selection(self, universe):
        '''
        allow the user to change the method for population selection.
        selection methods should be in codes/utilities/selections.py and generally are methods from deap.tools module

        setting default method to selNSGA2

        Going to select from hall_of_fame (if it exists) + population that aren't in hall_of_fame
        '''
        hall_of_fame_ids = []
        hall_of_fame_individuals= []
        if universe.population.hall_of_fame is not None:
            for indiv in universe.population.hall_of_fame.items:
                hall_of_fame_ids.append(indiv.id)
                # deepcopying from hall of fame so that in the future, any changes to this indiv,
                # won't affect the halloffame individuals.
                hall_of_fame_individuals.append(deepcopy(indiv))

        new_individuals = []
        for indiv in universe.population.population:
            if indiv.id not in hall_of_fame_ids:
                new_individuals.append(indiv)

        return selections.selNSGA2(hall_of_fame_individuals + new_individuals,
                                   k=self.pop_size,
                                   nd='standard')


    def parent_selection(self, universe):
        '''
        allow the user to change the method for parent selection.
        selection methods should be in codes/utilities/selections.py and generally are methods from deap.tools module

        setting default method to selTournamentDCD
        '''
        return selections.selTournamentDCD(universe.population.population, k=len(universe.population.population))


    def seed_with_previous_run(self, previous_universe_dir):
        '''
        we can set the genome_seeds attribute in Problem.__init__()
        or we can pass in a directory to find the pickled individuals when we call main.py

        this is useful for when we are running on a cluster like pace-ice with a compute time limit.
        at the end of a generation, we can start a new process for the next generation and reset the
        time limit...just have to pass the current output directory as input to the next run
        '''
        assert(os.path.exists(previous_universe_dir)), "Given 'previous_run' does not exist..."
        all_indiv = glob.glob(os.path.join(previous_universe_dir, "gen_*_indiv_*.pkl"))
        gen_dict = {}
        for indiv in all_indiv:
            # indiv should look like gen_###_indiv_hash.pkl
            gen = int(os.path.basename(indiv).split("_")[1])
            if gen in gen_dict:
                gen_dict[gen].append(indiv)
            else:
                gen_dict[gen] = [indiv]

        # now get the highest gen and all those individuals to genome_seeds
        largest_gen = max(list(gen_dict.keys()))
        self.genome_seeds = gen_dict[largest_gen]


    def postprocess_generation(self, universe):
        '''
        NOTE that this is not an abstractmethod because the user may choose not to do anything here

        at this point in universe, the population has been fully evaluated.
        currently it is at the end of the loop so population selection has already occured.
        that may change
        '''
        ezLogging.info("Post Processing Generation Run - pass")
        pass


    def postprocess_universe(self, universe):
        '''
        NOTE that this is not an abstractmethod because the user may choose not to do anything here

        the idea here is that the universe.run() is about to exit but before it does,
        we can export or plot things wrt the final population
        '''
        ezLogging.info("Post Processing Universe Run - default is to save population")
        save_things.save_population(self)


    def get_best_indiv(self, universe, ith_obj):
        '''
        return the index (ith indiv in population) who has the lowest score based
        on the ith objective
        '''
        best_score = np.inf
        best_index = None
        for ith_indiv, indiv in enumerate(universe.population.population):
            if indiv.fitness.values[ith_obj] < best_score:
                best_score = indiv.fitness.values[ith_obj]
                best_index = ith_indiv
        return best_index, best_score


    # Note to user: these last two methods are already defined
    def construct_block_def(self,
                            nickname: str,
                            shape_def: BlockShapeMeta_Abstract,
                            operator_def: BlockOperators_Abstract,
                            argument_def: BlockArguments_Abstract,
                            evaluate_def: BlockEvaluate_Abstract,
                            mutate_def: BlockMutate_Abstract,
                            mate_def: BlockMate_Abstract):
        return BlockDefinition(nickname,
                               shape_def,
                               operator_def,
                               argument_def,
                               evaluate_def,
                               mutate_def,
                               mate_def)


    def construct_individual_def(self,
                                 block_defs: List[BlockDefinition],
                                 mutate_def: IndividualMutate_Abstract,
                                 mate_def: IndividualMate_Abstract,
                                 evaluate_def: IndividualEvaluate_Abstract):
        self.indiv_def = IndividualDefinition(block_defs,
                                              evaluate_def,
                                              mutate_def,
                                              mate_def)

