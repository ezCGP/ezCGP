'''
root/problems/problem_syscosearch.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code.

Rules:
mention any assumptions made in the code or rules about code structure should go here

*See 'design' note in Problem class defined below
'''

### packages
import os, sys
import glob
import json
import numpy as np
import pandas as pd
import time, datetime
#import logging
import pdb

### sys relative to root dir
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from codes.utilities.custom_logging import ezLogging
from problems.problem_definition import ProblemDefinition_Abstract
from codes.factory import FactoryDefinition
#from data.data_tools import ezData
from data.data_tools import loader
# TODO pick out definitions needed; these are from example I copied from
from codes.block_definitions.shapemeta.block_shapemeta import BlockShapeMeta_Sysco
from codes.block_definitions.operators.block_operators import BlockOperators_SyscoSearch
from codes.block_definitions.arguments.block_arguments import BlockArguments_Auto
from codes.block_definitions.evaluate.block_evaluate import BlockEvaluate_MiddleBlock, BlockEvaluate_FinalBlock
from codes.block_definitions.mutate.block_mutate import BlockMutate_ArgValuesOnly
from codes.block_definitions.mate.block_mate import BlockMate_WholeOnly
from codes.individual_definitions.individual_mutate import IndividualMutate_RollOnEachBlock
from codes.individual_definitions.individual_mate import IndividualMate_RollOnEachBlock
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_SyscoSearch
from codes.block_definitions.block_definition import BlockDefinition
from misc import cx_search_lab_wrapper, seed_syscosearch


USE_BAYESIAN = True


class Problem(ProblemDefinition_Abstract):
    '''
    Design:
        I'm thinking of breaking up the hyperparameters into multiple blocks based off conceptual grouping; that way we can easily turn blocks on and off
        and mating will be super simple with just swaping entire blocks.
        Each block will have 1 primitive and 1 main node with no gene mating and only mutation is with the arguments
    '''
    def __init__(self):
        reset_experiment_result_log = False
        number_universe = 1
        factory = FactoryDefinition
        mpi = False
        genome_seeds = [['misc/IndivSeed_SyscoSearch_Originalv2_Seed/synonym_block_lisp.txt',
                         'misc/IndivSeed_SyscoSearch_Originalv2_Seed/product_description_block_lisp.txt',
                         'misc/IndivSeed_SyscoSearch_Originalv2_Seed/stocked_flag_block_lisp.txt',
                         'misc/IndivSeed_SyscoSearch_Originalv2_Seed/additional_boosts_block_lisp.txt',
                         'misc/IndivSeed_SyscoSearch_Originalv2_Seed/ranking_equation_block_lisp.txt'],
                        ['misc/IndivSeed_SyscoSearch_Originalv2_Seed/synonym_block_lisp.txt',
                         'misc/IndivSeed_SyscoSearch_Originalv2_Seed/product_description_block_lisp.txt',
                         'misc/IndivSeed_SyscoSearch_Originalv2_Seed/stocked_flag_block_v2_lisp.txt', # <- only this is different
                         'misc/IndivSeed_SyscoSearch_Originalv2_Seed/additional_boosts_block_lisp.txt',
                         'misc/IndivSeed_SyscoSearch_Originalv2_Seed/ranking_equation_block_lisp.txt']]
        if USE_BAYESIAN:
            population_size = 4
        else:
            READY = False
            if READY:
                genome_seeds = []
                # TODO build genome seeds or population from historical results
                experiment_historical_results, experiment_result_file, universe_timestamp = self.get_historical_log()
                for hyperparam_set in experiment_historical_results: # TODO this is just junky pseudo code
                    # go
                    seeded_indiv_folder = seed_syscosearch(*hyperparam_set)
                    seed = [os.path.join(seeded_indiv_folder, "%s_block_lisp.txt" % 'synonym'),
                            os.path.join(seeded_indiv_folder, "%s_block_lisp.txt" % 'product_description'),
                            os.path.join(seeded_indiv_folder, "%s_block_lisp.txt" % 'stocked_flag'),
                            os.path.join(seeded_indiv_folder, "%s_block_lisp.txt" % 'additional_boosts'),
                            os.path.join(seeded_indiv_folder, "%s_block_lisp.txt" % 'ranking_equation')]
                    genome_seeds.append(seed)
            else:
                # don't overwrite genome_seeds
                pass

            population_size = len(genome_seeds)

        hall_of_fame_flag = False
        super().__init__(population_size, number_universe, factory, mpi, genome_seeds, hall_of_fame_flag)
        self.relativeScoring = True # this will force universe to be instance of RelativePopulationUniverseDefinition() in main.py
        self.using_bayesian = USE_BAYESIAN # will be to set indiv_def for mate and mutate and for poulation_selection

        prob_mutate = 0.2
        num_mutants = 2
        prob_mate = 0.25
        synonym_operator_def = BlockOperators_SyscoSearch("synonym")
        synonym_block_def = self.construct_block_def(nickname = "synonym_block",
                                                    shape_def = BlockShapeMeta_Sysco,
                                                    argument_def = BlockArguments_Auto(synonym_operator_def, 1),
                                                    operator_def = synonym_operator_def,
                                                    evaluate_def = BlockEvaluate_MiddleBlock,
                                                    mutate_def = BlockMutate_ArgValuesOnly(prob_mutate, num_mutants),
                                                    mate_def = BlockMate_WholeOnly(prob_mate))
        synonym_block_def.freeze = True

        productdesc_operator_def = BlockOperators_SyscoSearch("product_desc")
        productdesc_block_def = self.construct_block_def(nickname = "product_description_block",
                                                    shape_def = BlockShapeMeta_Sysco,
                                                    argument_def = BlockArguments_Auto(productdesc_operator_def, 1),
                                                    operator_def = productdesc_operator_def,
                                                    evaluate_def = BlockEvaluate_MiddleBlock,
                                                    mutate_def = BlockMutate_ArgValuesOnly(prob_mutate, num_mutants),
                                                    mate_def = BlockMate_WholeOnly(prob_mate))
        productdesc_block_def.freeze = False

        stocked_operator_def = BlockOperators_SyscoSearch("stocked")
        stocked_block_def = self.construct_block_def(nickname = "stocked_flag_block",
                                                    shape_def = BlockShapeMeta_Sysco,
                                                    argument_def = BlockArguments_Auto(stocked_operator_def, 1),
                                                    operator_def = stocked_operator_def,
                                                    evaluate_def = BlockEvaluate_MiddleBlock,
                                                    mutate_def = BlockMutate_ArgValuesOnly(prob_mutate, num_mutants),
                                                    mate_def = BlockMate_WholeOnly(prob_mate))
        stocked_block_def.freeze = False

        addboosts_operator_def = BlockOperators_SyscoSearch("additional")
        addboosts_block_def = self.construct_block_def(nickname = "additional_boosts_block",
                                                    shape_def = BlockShapeMeta_Sysco,
                                                    argument_def = BlockArguments_Auto(addboosts_operator_def, 1),
                                                    operator_def = addboosts_operator_def,
                                                    evaluate_def = BlockEvaluate_MiddleBlock,
                                                    mutate_def = BlockMutate_ArgValuesOnly(prob_mutate, num_mutants),
                                                    mate_def = BlockMate_WholeOnly(prob_mate))
        addboosts_block_def.freeze = False
        
        rankeq_operator_def = BlockOperators_SyscoSearch("rank_equation")
        rankeq_block_def = self.construct_block_def(nickname = "ranking_equation_block",
                                                    shape_def = BlockShapeMeta_Sysco,
                                                    argument_def = BlockArguments_Auto(rankeq_operator_def, 1),
                                                    operator_def = rankeq_operator_def,
                                                    evaluate_def = BlockEvaluate_FinalBlock,
                                                    mutate_def = BlockMutate_ArgValuesOnly(prob_mutate, num_mutants),
                                                    mate_def = BlockMate_WholeOnly(prob_mate))
        rankeq_block_def.freeze = True

        if self.using_bayesian:
            # Going to use the same Individual Defs here BUT will change how we evolve the population at the Universe level
            # so check the code there
            indivdef_mutate = IndividualMutate_RollOnEachBlock
            indivdef_mate = IndividualMate_RollOnEachBlock
        else:
            indivdef_mutate = IndividualMutate_RollOnEachBlock
            indivdef_mate = IndividualMate_RollOnEachBlock

        self.construct_individual_def(block_defs = [synonym_block_def,
                                                    productdesc_block_def,
                                                    stocked_block_def,
                                                    addboosts_block_def,
                                                    rankeq_block_def],
                                      mutate_def = indivdef_mutate,
                                      mate_def = indivdef_mate,
                                      evaluate_def = IndividualEvaluate_SyscoSearch)

        self.construct_dataset()


    def set_optimization_goals(self):
        '''
        metric -> "result.term_metric_score"
        
        the metric is defined by the hyperparam file experiment_configs_v2.json in `metric` key:
            "expected_reciprocal_rank": {"maximum_relevance": 10, "k": 20}
        "a search relevance metric that measures how long a user is expected to take to find a relevant document." - Google
        bound by 0 and 1 -> https://notesonai.com/expected+reciprocal+rank 
        I think it is a maximization optimization based off the equation which has a factor of (1/k) where k is the rank of the 
        document, so the higher the rank, the later on in the document it is, and the worse it is, and ERR goes down.
        Higher rank, higher score, maximization.
        '''
        self.maximize_objectives = [True]
        self.objective_names = ["Expected Reciprocal Rank"] # will be helpful for plotting later


    def construct_block_def(self, nickname, *args, **kwargs):
        '''
        if we want to rely on some hash of the lips genome to record previously run/evaluated individuals, then we need to account
        for the 'stocking' operator that doesn't care about order of the arguments as they're passed in but rather sorts them and uses that order.
        '''
        if 'stock' in nickname:
            # then need custom ftn to make custom lisp
            return _custom_BlockDefinition(nickname, *args, **kwargs)
        
        else:
            return super().construct_block_def(nickname, *args, **kwargs)


    def construct_dataset(self):
        data_loader = loader.ezDataLoader_SyscoSearch()
        self.training_datalist, self.validating_datalist, self.testing_datalist = data_loader.load()


    def population_selection(self, universe):
        '''
        overwriting default method from ProblemDefinition to just avoid hall_of_fame completely and to keep 
        '''
        if self.using_bayesian:
            # we don't want to filter down like usual...just keep growing and building
            self.pop_size = len(universe.population.population)
            return universe.population.population

        else:
            super().popultion_selection(universe)


    def objective_functions(self, population):
        '''
        if using a RelativePopulationUniverseDefinition() instance of Universe() then our objective function will take a whole population instead of 
        just an individual...so we can wrap all the experimental configs into a list, write to file, and then run the simulation, grab the output, and assign 
        score to population
        '''
        # get historical results
        experiment_historical_results, experiment_result_file, universe_timestamp = self.get_historical_log()
        generation_timestamp = time.strftime("%Y%m%d-%H%M%S") # <- next best thing from generation number

        # get all individual experiment dicts/configs
        all_configs = []
        for indiv_material in population:
            indiv_experiment_config = indiv_material.output[0]
            try:
                assert isinstance(indiv_experiment_config, dict)
            except Exception as err:
                ezLogging.error(err)
                import pdb; pdb.set_trace()
            
            # lookup to see if we already ran and in historical log
            if not hasattr(indiv_material, 'hash'):
                self.assign_indiv_hash(indiv_material)
            if indiv_material.hash in experiment_historical_results:
                indiv_material.fitness.values = (experiment_historical_results[indiv_material.hash],)
                ezLogging.debug("Found Individual %s in previous generation..." % (indiv_material.id))
                continue

            all_configs.append(indiv_experiment_config)
    
        # write to json in cx_search_lab
        problem_output_folder = os.path.dirname(experiment_result_file)
        unique_tag = "%s_gen%s" % (universe_timestamp, generation_timestamp)
        all_experiments_config_file = os.path.join(problem_output_folder, 'experiment_config_ezcgp_%s.json' % (unique_tag))
        try:
            with open(all_experiments_config_file, 'w') as f:
                json.dump(all_configs, f, indent=2)
        except Exception as err:
            parse_for_type(all_configs[0]) # for debugging
            ezLogging.error(err)
            import pdb; pdb.set_trace()

        ### start 'analysis' step of cx_search_lab
        experiment_results = cx_search_lab_wrapper.main(all_experiments_config_file, unique_tag) # <-assume for now that we are still going to return a dict
    
        ### parse output file for true objective score and assign back to individual via indiv_material.fitness.values as tuple
        for indiv_material in population:
            if indiv_material.id in experiment_results:
                indiv_material.fitness.values = (experiment_results[indiv_material.id],)
                ezLogging.info("Scored Individual %s from current run: %.02f" % (indiv_material.id, indiv_material.fitness.values[0]))
            elif len(indiv_material.fitness.values) == 0:
                # fitness values should get initialized to empty ()
                # We shouldn't be at this point, so treat as error
                ezLogging.critical("Individual %s doesn't have a matching score/metric...Pronounced Dead" % indiv_material.id)
                indiv_material.dead = True
            else:
                # if filled in, then an individual evaluated from previous generation
                ezLogging.info("Scored Individual %s from previous generation with score: %.02f" % (indiv_material.id, indiv_material.fitness.values[0]))

        # log history for future reference
        self.log_results(population)


    def assign_indiv_hash(self, indiv_material):
        indiv_hash = []
        for block_def, block_material in zip(self.indiv_def, indiv_material):
            block_genome_hash = block_def.hash_lisp(block_material)
            block_genome_hash.replace('-','m')
            indiv_hash.append(block_genome_hash)
        indiv_material.hash = 'x'.join(indiv_hash)
        return indiv_material.hash


    def get_historical_log(self):
        '''
        NOTE that we would want to try and get universe.output_folder but don't want to change all the code just to pass in 'universe' variable just for this,
        so we are going to mimic the process from main.py that created this value
        '''
        problem_output_dir = os.path.join(dirname(dirname(realpath(__file__))),
                                          'outputs',
                                          os.path.splitext(os.path.basename(__file__))[0])
        experiment_results = glob.glob(os.path.join(problem_output_dir, 'experiment_results_*.json'))
        universe_timestamp = os.path.basename(max(glob.glob(os.path.join(problem_output_dir, '%i*' % datetime.datetime.now().year)) + 
                                                  glob.glob(os.path.join(problem_output_dir, 'testing-%i*' % datetime.datetime.now().year))))

        if (len(experiment_results)==0) or (self.reset_experiment_result_log):
            experiment_result_file = os.path.join(problem_output_dir, 'experiment_results_%s.json' % universe_timestamp)
            experiment_historical_results = {}
        else:
            experiment_result_file = max(experiment_results)
            with open(experiment_result_file, 'r') as f:
                experiment_historical_results = json.load(f)
        
        return experiment_historical_results, experiment_result_file, universe_timestamp


    def log_results(self, population):
        experiment_historical_results, experiment_result_file, _ = self.get_historical_log()
        
        for indiv_material in population:
            if not hasattr(indiv_material, 'hash'):
                self.assign_indiv_hash(indiv_material)
            
            if indiv_material.hash not in experiment_historical_results:
                experiment_historical_results[indiv_material.hash] = indiv_material.fitness.values[0]

        # overwrite historical results
        with open(experiment_result_file, 'w') as f:
            json.dump(experiment_historical_results, f)


    def check_convergence(self, universe):
        # TODO - how do we want to finish?
        ezLogging.info("still need to fillout check convergence"); pdb.set_trace()


        GENERATION_LIMIT = 5
        SCORE_MIN = 1e-1

        # only going to look at the first objective value which is rmse
        min_firstobjective_index = universe.pop_fitness_scores[:,0].argmin()
        min_firstobjective = universe.pop_fitness_scores[min_firstobjective_index,:-1]
        ezLogging.warning("Checking Convergence - generation %i, best score: %s" % (universe.generation, min_firstobjective))

        if universe.generation >= GENERATION_LIMIT:
            ezLogging.warning("TERMINATING...reached generation limit.")
            universe.converged = True
        if min_firstobjective[0] < SCORE_MIN:
            ezLogging.warning("TERMINATING...reached minimum scores.")
            universe.converged = True



class _custom_BlockDefinition(BlockDefinition):
    '''
    Need a custom class to help with editing lisp of the 'stock' block:
    since the operator in that block is independent of positional-index of input arguments, and rather sorts the values and assigns order that way,
    we want to do that step for the lisp so that we can get consistent hash and lisp from it.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_lisp(self, block_material):
        super().get_lisp(block_material)

        lisp = block_material.lisp[0].replace('[','').replace(']','').split(',')
        arg_vals = np.array(lisp[-3:], dtype=int)
        sorted_arg_vals = np.sort(arg_vals)[::-1]
        new_lisp = lisp[:-3] + list(sorted_arg_vals)
        # now follow the end of block_def.get_lisp()
        new_lisp_str = str(new_lisp)
        new_lisp_str = new_lisp_str.replace("'","").replace('"','').replace(" ", "")
        block_material.lisp[0] = new_lisp_str



def parse_for_type(mydict):
    '''
    had an issue with json.dump when writing out list of dicts
        "object of type int64 is not json serializable"
    so I wanted to print out all the types to see where the issue laid
    ...it was the argtypes with np.random.randomint or anything else that uses np
    '''
    for key, value in mydict.items():
        if isinstance(value, dict):
            parse_for_type(value)
        else:
            print(key, value, type(value))