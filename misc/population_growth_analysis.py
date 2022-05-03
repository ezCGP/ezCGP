'''
root/misc/population_growth_analysis.py
Response to Issue #242
Mating and Mutating params in ezCGP has been BARLEY looked into,
and I don't want to accidentally blow up my population size each generation.
This script should run several evolutions and build a histogram so we know
about how big of a population we can expect when evolving.
'''

### packages
import os
from copy import deepcopy
import time
import numpy as np

### sys relative to root dir
import sys
from os.path import dirname, realpath, join
sys.path.append(dirname(dirname(realpath(__file__))))
sys.path.append(join(dirname(dirname(realpath(__file__))), "problems")) # to easily import problem files

### absolute imports wrt root
from codes.universe import UniverseDefinition
import matplotlib.pyplot as plt


class MonteCarlo_PopulationGrowth_Universe(UniverseDefinition):
    '''
    don't necessarily need to mimic Universe class but it helps
    to show all these things are connected and where i got code from
    '''
    def __init__(self,
                 problem,
                 output_folder="/tmp/shoudlnt_matter",
                 random_seed=69):
        super().__init__(problem, output_folder, random_seed)


    def run(self,
            problem,
            simulation_count=1000,
            save=False):
        '''
        gonna run monte carlo sim and make the plot
        '''
        # Init Population
        self.population = self.factory.build_population(problem, problem.pop_size, 0, 1)

        # Copy for Resetting later
        orig_population = deepcopy(self.population.population)

        population_count = []
        from_mating = []
        from_mutating = []
        for _ in range(int(simulation_count)):
            if _%100==0:
                print("Status Update: %i/%i" % (_, simulation_count))

            # SCORE
            # ...helps for parent selection in mating
            for indiv in self.population.population:
                fake_fitness = np.random.random(size=(len(problem.maximize_objectives)))
                indiv.fitness.values = tuple(fake_fitness)

            # RANK
            # ...parent selection needs crowding_dist attr filled
            self.population_selection(problem)

            # MATE
            current_count = len(self.population.population)
            self.mate_population(problem)
            #from_mating.append(len(self.population.population)-current_count)
            from_mating.append(len(self.population.population)/current_count)

            # MUTATE
            current_count = len(self.population.population)
            self.mutate_population(problem)
            #from_mutating.append(len(self.population.population)-current_count)
            from_mutating.append(len(self.population.population)/current_count)
            population_count.append(len(self.population.population))

            # RESET
            self.population.population = deepcopy(orig_population)

        population_count = np.array(population_count)
        from_mating = np.array(from_mating)
        from_mutating = np.array(from_mutating)

        # PLOT
        fig, axes = plt.subplots(3, 1, figsize=(5,14), sharex=True)
        binsize = 1
        for i, (counts, title) in enumerate(zip([population_count, from_mating, from_mutating],
                                                ["Population Growth", "...from Mating Only", "...from Mutating Only"])):
            min_bin = counts.min()
            max_bin = counts.max() + binsize
            axes[i].hist(counts, bins=np.arange(min_bin, max_bin))
            if i == 0:
                axes[i].set_title("%s\nAve %.1f" % (title, counts.mean()))
            else:
                axes[i].set_title("%s\nAve Multiplier %.1f" % (title, counts.mean()))

        if save:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            image_name = "./test_area/offspring_count-pop%03i-%s.png" % (len(orig_population), time_str)
            plt.savefig(image_name)        
        else:
            plt.show()

        try:
            plt.close()
        except Exception:
            pass



if __name__ == "__main__":
    '''
    example:
    $ python misc/population_growth_analysis.py -p problem_simgan --c 10000 --save

    Note that it doesn't save with the problem name...too much work, sorry
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--problem",
                        type = str,
                        required = True,
                        help = "pick which problem class to import")
    parser.add_argument("-c", "--count",
                        type = int,
                        required = False,
                        default = 5000,
                        help = "monte carlo simulation count.")
    parser.add_argument("-s", "--save",
                        action = "store_const",
                        const = True,
                        default = False,
                        help = "flag to save the final plot to test_area")

    args = parser.parse_args()

    # Import Problem File
    if args.problem.endswith('.py'):
        problem_filename = os.path.basename(args.problem)
    else:
        problem_filename = os.path.basename(args.problem + ".py")

    problem_module = __import__(problem_filename[:-3]) #remove the '.py' from filename
    problem = problem_module.Problem()
    universe = MonteCarlo_PopulationGrowth_Universe(problem)
    universe.run(problem, args.count, args.save)