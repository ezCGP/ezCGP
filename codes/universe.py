'''
root/code/universe.py

Overview:
overview of what will/should be in this file and how it interacts with the rest of the code

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
import numpy as np
from typing import List
import importlib
import time
from copy import deepcopy
from mpi4py import MPI

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems.problem_definition import ProblemDefinition_Abstract
from codes.utilities.custom_logging import ezLogging
from post_process import save_things



class UniverseDefinition():
    '''
    TODO
    '''
    def __init__(self,
                 problem: ProblemDefinition_Abstract,
                 output_folder: str,
                 random_seed: int):
        '''
        electing to keep Problem class separate from Universe class...
        if we ever need a Problem attribute, just pass in the instance of Problem as an arguement
        ...as opposed to having a copy of the Problem instance as an attribute of Universe
        '''
        self.adjust_pop_size(problem, [4]) # selections.selTournamentDCD requires multiples of 4
        self.factory = problem.Factory()
        # moving init of population in run()...move back here if it doesn't work
        #self.population = self.factory.build_population(problem.indiv_def, problem.pop_size)
        #self.problem = problem #did we really need the problem as attr? cleaner to keep them separate, right?
        self.output_folder = output_folder
        self.random_seed = random_seed
        self.converged = False

        # match a few attributes found in MPIUniverseDefinition
        self.node_number = MPI.COMM_WORLD.Get_rank() #0 if not mpi
        self.node_count = MPI.COMM_WORLD.Get_size() #1 if not mpi


    def adjust_pop_size(self,
                        problem: ProblemDefinition_Abstract,
                        multiples_of: List[int]):
        '''
        in certain cases, we may need to have our population size be a multiple of something so that the code doesn't break.
        for example, if we use some tournament selection, the poulation will likely need to be a multiple of 2 or 4.
        or if we use mpi, then we would want our pop size to be a multiple of the number of nodes we are using out of convenience.
        '''
        original_size = problem.pop_size
        possible_size = deepcopy(original_size)
        satisfied = False
        direction = "down"
        while not satisfied:
            # loop until we have a pop_size that is a multiple of all ints in the list
            for mult in multiples_of:
                if possible_size%mult != 0:
                    # not a multiple...not satisfied
                    if direction == "down":
                        possible_size -= possible_size%mult
                    else:
                        possible_size += (mult-possible_size%mult)
                    satisfied = False
                    break
                else:
                    # it is a multiple...so far, we are satisfied
                    satisfied = True

            if possible_size <= 0:
                # then we failed...try changing direction
                print(possible_size)
                satisfied = False
                direction = "up"
                possible_size = deepcopy(original_size)
        if possible_size != original_size:
            ezLogging.warning("Changing problem's population size from %i to %i to be multiples of %s" % (original_size, possible_size, multiples_of))
        problem.pop_size = possible_size


    def parent_selection(self, problem: ProblemDefinition_Abstract):
        '''
        moved to problem class so user can easily customize selection method.
        selection methods should be in codes/utilities/selections.py and generally are methods from deap.tools module.
        method should return a list of parents
        '''
        return problem.parent_selection(self)


    def mate_population(self, problem: ProblemDefinition_Abstract):
        '''
        do a ranking/sorting of parents, then pair them off, mate the pairs, return and add the children
        '''
        start_time = time.time()
        children = []
        mating_list = self.parent_selection(problem)
        for ith_indiv in range(0, len(mating_list), 2):
            parent1 = mating_list[ith_indiv]
            parent2 = mating_list[ith_indiv+1]
            children += problem.indiv_def.mate(parent1, parent2)
        self.population.add_next_generation(children)
        ezLogging.info("Node %i - Mating took %.2f minutes" % (self.node_number, (time.time()-start_time)/60))



    def mutate_population(self, problem: ProblemDefinition_Abstract):
        '''
        super simple...just loop through am call mutate. at the block level is where it decides to mutate or not
        '''
        start_time = time.time()
        mutants = []
        for individual in self.population.population:
            mutants += problem.indiv_def.mutate(individual)
        self.population.add_next_generation(mutants)
        ezLogging.info("Node %i - Mutation took %.2f minutes" % (self.node_number, (time.time()-start_time)/60))



    def evolve_population(self, problem: ProblemDefinition_Abstract):
        '''
        TODO
        '''
        # MATE
        self.mate_population(problem)
        ezLogging.info("Population size after Mating: %i" % (len(self.population.population)))

        # MUTATE
        self.mutate_population(problem)
        ezLogging.info("Population size after Mutating: %i" % (len(self.population.population)))


    def evaluate_score_population(self, problem: ProblemDefinition_Abstract, compute_node: int=None):
        '''
        TODO
        '''
        self.pop_fitness_scores = []
        self.pop_individual_ids = []
        ezLogging.warning("Evaluating Population of size %i" % (len(self.population.population)))
        for indiv in self.population.population:
            # EVALUATE
            problem.indiv_def.evaluate(indiv, problem.training_datalist, problem.validating_datalist)
            # SCORE
            problem.objective_functions(indiv)
            self.pop_fitness_scores.append(indiv.fitness.values)
            # ATTACH ID
            self.pop_individual_ids.append(indiv.id)
        self.pop_fitness_scores = np.array(self.pop_fitness_scores)
        self.pop_individual_ids = np.array(self.pop_individual_ids)


    def update_hall_of_fame(self):
        '''
        wrapper function to call the deap.tools.HallOfFame.update(population) method
        '''
        self.population.update_hall_of_fame()


    def population_selection(self, problem: ProblemDefinition_Abstract):
        '''
        moved to problem class so user can easily customize selection method.
        selection methods should be in codes/utilities/selections.py and generally are methods from deap.tools module.
        method should alter the self.population.population attr
        '''
        problem.population_selection(self)


    def check_convergence(self, problem: ProblemDefinition_Abstract):
        '''
        Should update self.converged
        '''
        problem.check_convergence(self)


    def postprocess_generation(self, problem: ProblemDefinition_Abstract):
        '''
        Just a wrapper to problem.postprocess_universe()

        Could decide to save stats about the generation pareto front etc or even save individuals.
        May decide to move before self.population_selection() so that we can get stats on the whole pop before
        any trimming...depends on what things we're interested in. Right now I made no decision on what to collect
        so it's at the end of the generation loop
        '''
        problem.postprocess_generation(self)


    def postprocess_universe(self, problem: ProblemDefinition_Abstract):
        '''
        Wrapper to problem.postprocess_universe()

        Provides an option for anything we want to do with the universe + population now that we reached the
        complete end of the evolutionary cycle.
        '''
        problem.postprocess_universe(self)


    def run(self, problem: ProblemDefinition_Abstract):
        '''
        assumes a population has only been created and not evaluated/scored
        '''
        self.generation = 0
        self.population = self.factory.build_population(problem, problem.pop_size, self.node_number, self.node_count)
        self.evaluate_score_population(problem)
        self.update_hall_of_fame()
        self.population_selection(problem)
        self.check_convergence(problem)
        self.postprocess_generation(problem)
        while not self.converged:
            self.generation += 1
            ezLogging.warning("Starting Generation %i" % self.generation)
            self.evolve_population(problem)
            self.evaluate_score_population(problem)
            self.update_hall_of_fame()
            self.population_selection(problem)
            self.check_convergence(problem)
            self.postprocess_generation(problem)
        self.postprocess_universe(problem)



class MPIUniverseDefinition(UniverseDefinition):
    '''
    use mpi for multiprocessing on other cpu nodes
    '''
    def __init__(self,
                 problem: ProblemDefinition_Abstract,
                 output_folder: str,
                 random_seed: int):
        '''
        TODO
        '''
        #from mpi4py import MPI # gave in and going to use everywhere no matter what
        #globals()['MPI'] = MPI
        # adjust size to be mult of 4 for tournselection
        # and adjust to be 2*number of nodes for mpi so that when we divide up the population, there is
        # an even number of individuals for each node...this way, we can do parent selection beffore splitting
        # and the parent-pairs won't get split
        self.adjust_pop_size(problem, [4, 2*MPI.COMM_WORLD.Get_size()])
        super().__init__(problem, output_folder, random_seed)


        ''' cannot import MPI as an attribute since it's a subpackage!
        # globally import mpi
        mdl = importlib.import_module("mpi4py")
        #
        if '__all__' in mdl.__dict__: # true here it seems
            names = mdl.__dict__['__all__']
        else:
            names = [x for x in mdl.__dict__ if not x.startswith('_')]
        globals().update({name: getattr(mdl, name) for name in names})'''


    def mpi_mate_population(self, problem: ProblemDefinition_Abstract):
        '''
        the only difference here with mate_population() [<-no mpi] is that we do parent_selection before hand
        because by this point we already split and scattered the population to each node and we'd prefer to do
        parent selection amoung the whole population

        earlier at __init__ we should have adjusted pop size to be a multiple of 4 and multiples of our node count
        so this should gaurentee that parents do not get split up when we split+scatter to the different nodes from root=0
        '''
        start_time = time.time()
        children = []
        mating_list = self.population.population
        ezLogging.debug("here mating %i" % (len(mating_list)))
        for ith_indiv in range(0, len(mating_list), 2):
            parent1 = mating_list[ith_indiv]
            parent2 = mating_list[ith_indiv+1]
            children += problem.indiv_def.mate(parent1, parent2)
        self.population.add_next_generation(children)
        ezLogging.info("Node %i - Mating took %.2f minutes" % (self.node_number, (time.time()-start_time)/60))


    def mpi_evolve_population(self, problem: ProblemDefinition_Abstract):
        '''
        mpi wrapper around UniverseDefinition.evolve_population
        '''
        ### MATE
        # NOTE we should have already done parent selection before split+scatter
        self.mpi_mate_population(problem)

        ### MUTATE
        self.mutate_population(problem)


    def mpi_evaluate_score_population(self, problem: ProblemDefinition_Abstract):
        '''
        a wrapper method to handle the scatter+gather to evaluate

        assume that at this point we have already split + scattered the population

        then we want to evaluate, wait for everything to finish with Barrier, and then recollect
        '''
        self.evaluate_score_population(problem)


    def split_scatter_population(self, problem: ProblemDefinition_Abstract, parent_selection=False):
        '''
        small method where we assume that the population is a full list of individualmaterial
        and we want to split it up into list of list and then scatter each of those lists to a node

        added parent selection here as well to rank/sort into pairs for mating later...doing it here as a full population
        rather than later as a subpopulation
        '''
        if self.node_number == 0:
            if parent_selection:
                self.population.population = self.parent_selection(problem) #id prefer to do this before we split and scatter
            self.population.split_population(self.node_count)
        self.population.population = MPI.COMM_WORLD.scatter(self.population.population, root=0)
        MPI.COMM_WORLD.Barrier()


    def gather_population(self):
        '''
        each node has a subpopulation and now we want to collect all the subpops into a single list
        and bring them to our root node
        '''
        MPI.COMM_WORLD.Barrier()
        subpops = MPI.COMM_WORLD.gather(self.population.population, root=0)
        if self.node_number == 0:
            self.population.merge_subpopulations(subpops)


    def run(self, problem: ProblemDefinition_Abstract):
        '''
        1. Split Populatio n into subpopulation such that number of sup-pops == number of CPUs
        Loop:
        2. Scatter Sub-population from CPU 0 to each of the cpu
        3. Evolve Sub-Population on each CPU
        4. Evaluate Sub-population on each CPU
        5. Gather all the sub-population to CPU 0 (Master CPU)
        6. Perform MPI population selection on CPU 0
            - This produce a new array of sub-population
        7. Check convergence on CPU 0
        8. Broadcast Convergence status to all CPUs
        Repeat Step 2
        '''

        self.generation = 0
        # start split until after evaluate
        self.population = self.factory.build_population(problem,
                                                        problem.pop_size//self.node_count, #each node make their own fraction of indiv...but how does seeding work, are they dups?
                                                        self.node_number,
                                                        self.node_count)
        ezLogging.warning("START")
        import tensorflow as tf
        ezLogging.warning("A")
        gpu_count = tf.config.experimental.list_physical_devices('GPU')
        ezLogging.warning("B")
        import os
        cmd = "hostname -I | awk '{print $1}'"
        ip_address = os.popen(cmd).read()[:-1]
        cmd = "uname -n"
        hostname = os.popen(cmd).read()[:-1]
        ezLogging.warning("C")
        ezLogging.critical("Node %i - gpu count: %s at %s on %s" % (self.node_number, gpu_count, ip_address, hostname))
        # TODO verify seeding
        # TODO verify that we are handling different indiv_id's for pop creation
        # TODO verify ezLogging
        self.mpi_evaluate_score_population(problem)
        self.gather_population()
        if self.node_number == 0:
            self.update_hall_of_fame()
            self.population_selection(problem)
            self.check_convergence(problem)
            self.postprocess_generation(problem)
        while not self.converged:
            self.generation += 1
            self.split_scatter_population(problem, parent_selection=True)
            self.mpi_evolve_population(problem)
            self.mpi_evaluate_score_population(problem)
            self.gather_population()
            if self.node_number == 0:
                self.update_hall_of_fame()
                self.population_selection(problem)
                self.check_convergence(problem)
                self.postprocess_generation(problem)
            # if converged goes to True then we want all nodes to have that value changed
            MPI.COMM_WORLD.Barrier()
            self.converged = MPI.COMM_WORLD.bcast(self.converged, root=0)
        if self.node_number == 0:
            self.postprocess_universe(problem)



class RelativePopulationUniverseDefinition(UniverseDefinition):
    '''
    Defines a Universe specifically for problems where the individuals are judged by a relative performance metric instead of an absolute performance metric
    '''
    def __init__(self,
                 problem: ProblemDefinition_Abstract,
                 output_folder: str,
                 random_seed: int):
        ezLogging.info("Using Relative Population Universe")
        super().__init__(problem, output_folder, random_seed)



    def evaluate_score_population(self, problem: ProblemDefinition_Abstract, compute_node: int=None):
        '''
        Evaluates and scores the population
        '''
        self.pop_fitness_scores = []
        self.pop_individual_ids = []

        # EVALUATE
        ezLogging.info("Evaluating Population of size %i" % (len(self.population.population)))
        for indiv in self.population.population:
            problem.indiv_def.evaluate(indiv, problem.training_datalist, problem.validating_datalist)

        # SCORE
        ezLogging.info("Scoring Population of size %i" % (len(self.population.population)))
        problem.objective_functions(self.population)

        # GET SCORES AND IDS
        for indiv in self.population.population:
            self.pop_fitness_scores.append(indiv.fitness.values)
            self.pop_individual_ids.append(indiv.id)

        self.pop_fitness_scores = np.array(self.pop_fitness_scores)
        self.pop_individual_ids = np.array(self.pop_individual_ids)