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
import logging

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems.problem_definition import ProblemDefinition_Abstract
from codes.utilities import selections




class UniverseDefinition():
    '''
    TODO
    '''
    def __init__(self,
                 problem: ProblemDefinition_Abstract,
                 output_folder: str):
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
        self.converged = False


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
            logging.warning("Changing problem's population size from %i to %i to be multiples of %s" % (original_size, possible_size, multiples_of))
        problem.pop_size = possible_size


    def parent_selection(self):
        '''
        TODO
        '''
        return selections.selTournamentDCD(self.population.population, k=len(self.population.population))


    def mate_population(self, problem: ProblemDefinition_Abstract, compute_node: int=None):
        '''
        do a ranking/sorting of parents, then pair them off, mate the pairs, return and add the children
        '''
        start_time = time.time()
        children = []
        mating_list = self.parent_selection()
        for ith_indiv in range(0, len(mating_list), 2):
            parent1 = mating_list[ith_indiv]
            parent2 = mating_list[ith_indiv+1]
            children += problem.indiv_def.mate(parent1, parent2)
        self.population.add_next_generation(children)
        logging.info("Node %i - Mating took %.2f minutes" % (compute_node, (time.time()-start_time)/60))



    def mutate_population(self, problem: ProblemDefinition_Abstract, compute_node: int=None):
        '''
        super simple...just loop through am call mutate. at the block level is where it decides to mutate or not
        '''
        start_time = time.time()
        mutants = []
        for individual in self.population.population:
            mutants += problem.indiv_def.mutate(individual)
        self.population.add_next_generation(mutants)
        logging.info("Node %i - Mutation took %.2f minutes" % (compute_node, (time.time()-start_time)/60))



    def evolve_population(self, problem: ProblemDefinition_Abstract):
        '''
        TODO
        '''
        # MATE
        self.mate_population(problem)

        # MUTATE
        self.mutate_population(problem)


    def evaluate_score_population(self, problem: ProblemDefinition_Abstract, compute_node: int=None):
        '''
        TODO
        '''
        self.fitness_scores = []
        for indiv in self.population.population:
            # EVALUATE
            problem.indiv_def.evaluate(indiv, problem.data.x_train)
            # SCORE
            problem.objective_functions(indiv)
            # ATTACH TO ID
            id_scores = indiv.fitness.values + (indiv.id,)
            self.fitness_scores.append(id_scores)
        self.fitness_scores = np.array(self.fitness_scores)


    def population_selection(self):
        '''
        TODO
        '''
        self.population.population, _ = selections.selNSGA2(self.population.population, self.population.pop_size, nd='standard')


    def check_convergence(self, problem_def: ProblemDefinition_Abstract):
        '''
        Should update self.converged
        '''
        problem_def.check_convergence(self)


    def postprocess_generation(self, problem_def: ProblemDefinition_Abstract):
        '''
        Just a wrapper to problem.postprocess_universe()

        Could decide to save stats about the generation pareto front etc or even save individuals.
        May decide to move before self.population_selection() so that we can get stats on the whole pop before
        any trimming...depends on what things we're interested in. Right now I made no decision on what to collect
        so it's at the end of the generation loop
        '''
        problem_def.postprocess_generation(self)


    def postprocess_universe(self, problem_def: ProblemDefinition_Abstract):
        '''
        Wrapper to problem.postprocess_universe()

        Provides an option for anything we want to do with the universe + population now that we reached the 
        complete end of the evolutionary cycle.
        '''
        problem_def.postprocess_universe(self)


    def run(self, problem: ProblemDefinition_Abstract):
        '''
        assumes a population has only been created and not evaluatedscored
        '''
        self.generation = 0
        self.population = self.factory.build_population(problem.indiv_def, problem.pop_size)
        self.evaluate_score_population(problem)
        self.population_selection()
        while not self.converged:
            self.generation += 1
            self.evolve_population(problem)
            self.evaluate_score_population(problem)
            self.population_selection()
            self.check_convergence(problem)
            self.postprocess_generation(problem)
        self.postprocess_universe(problem)



class MPIUniverseDefinition(UniverseDefinition):
    '''
    use mpi for multiprocessing on other cpu nodes
    '''
    def __init__(self,
                 problem: ProblemDefinition_Abstract,
                 output_folder: str):
        '''
        TODO
        '''
        from mpi4py import MPI
        globals()['MPI'] = MPI
        self.adjust_pop_size(problem, [4, MPI.COMM_WORLD.Get_size()])
        super().__init__(problem, output_folder)

        ''' cannot import MPI as an attribute since it's a subpackage!
        # globally import mpi
        mdl = importlib.import_module("mpi4py")
        #
        if '__all__' in mdl.__dict__: # true here it seems
            names = mdl.__dict__['__all__']
        else:
            names = [x for x in mdl.__dict__ if not x.startswith('_')]
        globals().update({name: getattr(mdl, name) for name in names})'''


    def mpi_evolve_population(self, problem: ProblemDefinition_Abstract, comm_world):
        '''
        mpi wrapper around UniverseDefinition.evolve_population
        '''
        ### MATE
        # only on main node for now
        if comm_world.Get_rank() == 0:
            self.mate_population(problem)

        ### MUTATE
        # do we gain anything by splitting for mutation? is it not fast enough already?
        self.population.population = comm_world.scatter(self.population.population, root=0)
        self.mutate_population(problem, comm_world.Get_rank())
        comm_world.Barrier() #what does this do?
        subpops = comm_world.gather(self.population.population, root=0)
        if comm_world.Get_rank() == 0:
            self.population.merge_subpopulations(subpops)


    def mpi_evaluate_score_population(self, problem: ProblemDefinition_Abstract, comm_world):
        '''
        a wrapper method to handle the scatter+gather to evaluate
        '''
        # check if we have a already split into subpops
        if isinstance(self.population.population[0], list):
            # then we can assume we already split into subpops
            pass
        else:
            if comm_world.Get_rank() == 1:
                self.population.split_population(comm_world.Get_size())
            comm_world.Barrier()
        # i think scatter returns the subpopulation for the specific node (the rank^th node)
        self.population.population = comm_world.scatter(self.population.population, root=0) # why assign it to itself?
        self.evaluate_score_population(problem, comm_world.Get_rank())
        comm_world.Barrier() #what does this do?
        subpops = comm_world.gather(self.population.population, root=0) # returns list of list of individuals
        if comm_world.Get_rank() == 0:
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
        comm = MPI.COMM_WORLD
        size = comm.Get_size()  # number of CPUs #not used anymore
        rank = comm.Get_rank()  # this CPU's rank

        self.generation = 0
        # only have main node create the population
        #if rank == 0:
        # START with split populations
        self.population = self.factory.build_population(problem.indiv_def, problem.pop_size//size) #each node make their own fraction of indiv...but how does seeding work, are they dups?
        # TODO verify seeding
        # TODO verify that we are handling different indiv_id's for pop creation
        # TODO verify logging
        self.mpi_evaluate_score_population(problem, comm)
        self.population_selection()
        while not self.converged:
            self.generation += 1
            self.mpi_evolve_population(problem, comm)
            self.mpi_evaluate_score_population(problem, comm)
            if rank == 0:
                self.population_selection()
                self.check_convergence(problem)
                self.postprocess_generation(problem)
                self.population.split_population(size)
            # if converged goes to True then we want all nodes to have that value changed
            self.converged = comm.bcast(self.converged, root=0)
        if rank == 0:
            self.postprocess_universe(problem)