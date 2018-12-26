### cgp_generator.py

# packages:
from copy import copy, deepcopy
from scoop import futures
import scoop
import pickle
import gc
import sys
import math

# other python scripts:
from configuration import *
from arguments import *
#from operators import *
from individual import *
#from fromExternal.emo import selNSGA2
from fromExternal.hv import HyperVolume
from evaluation_log import my_logging



from fromExternal.emo import selNSGA2, selTournamentDCD #do we need to do this?
multiObj = selNSGA2


def generate(population, ind_queue, gens, global_inputs, f):
    '''
    Take in a new population ready for producing new offspring.
    The population has already gone through some sort of multiobjective ranking system (like NSGA2),
    but not tournament selection for mating.

    Starts with Mating, then Mutation.

    Offspring are added to the ind_queue to be evaluated on the clusters
    '''
    to_write = my_logging("new generation", [gens])
    f.write(to_write)

    if compute == "scoop":
        gen_id = os.getpid()
        gen_mem = int(os.popen('ps -p %d -o %s | tail -1' % (
                                    gen_id, 'rss')).read()) / 1024.0
        gen_file = "/tmp/rt_gen_mem_%s_%s_%s" % (gens, gen_id, gen_mem)
        open(gen_file, "a").close()

        init_mem = int(os.popen('ps -p %d -o %s | tail -1' % (
                                    init_id, 'rss')).read()) / 1024.0   
        init_file = "/tmp/rt_init_mem_%s_%s_%s" % (gens, init_id, init_mem)
        open(init_file, "a").close()
    else:
        pass



    # list of parents to mate with:
    parents = selTournamentDCD(population, k=pop_size)
    for i in range(0,len(parents),2): #iterate by 2
        if mate_type in ["normal", "subgraph"] and random.random() <= mate_rate: # then mate
            crossover = mate_type
        else: # dont mate
            crossover = "none"

        parent1 = deepcopy(parents[i])
        parent2 = deepcopy(parents[i+1])
        offsprings = parent1.mate(parent2, crossover, keep_both_offspring)

        for o, offspring in enumerate(offsprings):

            # prep offspring to send off for evaluation
            offspring.id = "%s_%s_%s" %(gens, int(i/2+1), o+1)
            offspring.to_write = my_logging("mate", [gens, int(i/2+1), o+1, crossover])
            offspring.checkTypes()
            offspring.findActive()
            #offspring.to_write = my_logging("mate", [gens, int(i/2+1), o+1, crossover])
            
            # evaluate block
            if compute == "scoop":
                ind_queue.append(futures.submit(evaluate, offspring, global_inputs))
            elif compute == "local":
                ind_queue.append(evaluate(offspring, global_inputs))
            else:
                pass
            
            # reorder the genome
            if ordering == "reorder":
                    offspring.reorder()
            else:
                pass
            
            # begin mutating process
            for m in range(mutant_count):
            #if random.random() <= ind_mut_rate: # then mutate!
                #for m in range(mutant_count):
                if random.random() <= ind_mut_rate: # then mutate!
                    # copy all offspring attributes to the mutants, then mutate
                    mutant = deepcopy(offspring)
                    # reset the following attribute and then mutate
                    #mutant.active_node_changed = False...do it in mutate()
                    mutant.mutate()
                    # normally here we would check if an active node changed ("skip"),
                    # but since evaluation is stalled until later, we can't pass on fitness
                    # and evaluation values if no active node changed.
                    mutant.id = "%s_%s_%s_%s" %(gens, int(i/2+1), o+1, m+1)
                    mutant.to_write = my_logging("mutate", [gens, int(i/2+1), o+1, m+1])
                    mutant.checkTypes()
                    mutant.findActive()
                    #mutant.to_write = my_logging("mutate", [gens, int(i/2+1), o+1, m+1])

                    # evaluate block
                    if compute == "scoop":
                        ind_queue.append(futures.submit(evaluate, mutant, global_inputs))
                    elif compute == "local":
                        ind_queue.append(evaluate(mutant, global_inputs))
                    else:
                        pass

                else: #don't mutate
                    pass
    return ind_queue, f

### RODD ADDED:
def change_mate_rate(front):
    if adaptable:
        # note that pareto front starts at size 1 not 0
        m = -2*set_mate_rate/(pop_size-1) # *2 to make it decrease faster
        mate_rate = m * (len(front) - 1) + set_mate_rate
        # a negative mate_rate is okay...just means that it won't mate
    else:
        mate_rate = set_mate_rate
    #print("front size2:", len(fronts[-1]), mate_rate)
    return mate_rate


# define main population generator below:
def start_run(run, seed, global_inputs, test_inputs):
    '''
    Here is what happens in 1 run:
        * initialize genome and arguements for an individual
        * fill a population with these individuals
        * evaluate genome and calculate fitness of initial parent and population
        * rank and sort, then perform tournament selection for mating
        * decide whether to mate or not and pass the parents or the offspring
        * mutate the parents or offsprings genome and arguement list
        * evaluate genome and calculate fitness of all offspring
        * rank the population and trim down in size; calculate population fitness
        * repeat
    Then test for model convergence.

    start_run() initializes the population and the states, and then sends the population to
    generate() to organize and handle all the generation of offspring.

    Once generator finishes, we start to collect the individuals with fitness calculated and add them
    back to the population. If that population reaches a minimum count and if the pareto front has not
    yet converged, it is sent to generate another set of offsrping. This way, it doesn't have to wait for
    all the individuals to finish [only for parallelized computation].
    '''
    
    random.seed(seed)
    np.random.seed(seed)


    # initialize new file for writing the detailed evaluation log:
    file = "outputs/evaluation_log_%s.txt" % run
    f, to_write = my_logging(type="initialize", to_write=[file,seed])

    # initialize arguement list
    global numArgs
    numArgs = 0
    argument_skeleton, numArgs = build_arg_skeleton(set_numArgs) #in arguments.py

    ######
    #import collections
    #counter = collections.Counter()
    argDict = {}
    for arg in argument_skeleton:
        arg_type = type(arg).__name__ #saved as str
        if arg_type in argDict:
            argDict[arg_type] += 1
        else:
            argDict[arg_type] = 1

    print(argDict)
    #wait = input("check args real quick")

    if compute == "scoop":
        global init_id
        init_id = os.getpid()
        init_mem = int(os.popen('ps -p %d -o %s | tail -1' % (
            init_id, 'rss')).read()) / 1024.0
        init_file = "/tmp/rt_init_mem_0_%s_%s" % (init_id, init_mem)
        open(init_file, "a").close()
    else:
        pass


    # initialize the population
    population = []
    ind_queue = []
    for i in range(pop_size):
        individual = Individual(argument_skeleton, numArgs)
        individual.id = "1_%s" % (i)
        individual.checkTypes()
        individual.findActive()
        individual.to_write = to_write + str(i)+ ":\n"
        
        # evaluate block
        if compute == "scoop":
            ind_queue.append(futures.submit(evaluate, individual, global_inputs))
            asfile = "/tmp/rt_0_%s_addedtoQ" % (i)
            open(asfile, "a").close()
        elif compute == "local":
            ind_queue.append(evaluate(individual, global_inputs))
        else:
            pass

    if compute == "scoop":
        open("/tmp/rt_0_allAdded2Q", "a").close()
        scoop._control.execQueue.updateQueue()
        open("/tmp/rt_0_shouldStartEval", "a").close()
    else:
        pass

    # collect the evaluated fitnesses
    while len(ind_queue) > 0:
        #open("/tmp/rt_z_loopcheck1", "a").close()
        time.sleep(60)
        i = 0
        #open("/tmp/rt_z_loopcheck2", "a").close()
        while i < len(ind_queue):
            ind_future = ind_queue[i]
            if compute == "scoop":
                if ind_future.done():
                    individual = ind_future.result()
                    population.append(individual)

                    #init_mem = int(os.popen('ps -p %d -o %s | tail -1' % (
                    #            init_id, 'rss')).read()) / 1024.0
                    #ram_file = "/tmp/rt_ram_%s-%s-%s_%s_%s_%s" % (i, len(ind_queue), min_queue_size, sys.getsizeof(individual), sys.getsizeof(population), init_mem)
                    #open(ram_file, 'a').close()

                    f.write(individual.to_write)
                    ind_queue.pop(i)
                else:
                    #init_mem = int(os.popen('ps -p %d -o %s | tail -1' % (
                    #            init_id, 'rss')).read()) / 1024.0
                    #ram_file = "/tmp/rt_ram_%s-%s-%s_%s_%s_%s_NOTDONE" % (i, len(ind_queue), min_queue_size, sys.getsizeof(individual), sys.getsizeof(population), init_mem)
                    #open(ram_file, 'a').close()
                    i+=1
            elif compute == "local":
                population.append(ind_future)
                f.write(ind_future.to_write)
                ind_queue.pop(i)
            gc.collect()












    print("\n\n\n\n\nPOPULATION HAS BEEN INITIALIZED. READY TO BEGIN EVOLUTION!!!!!\n", len(ind_queue), len(population),"\n\n",flush=True)

    #for future_ind in ind_queue:
    #    if compute == "scoop":
    #        individual =  future_ind.result()
    #    elif compute == "local":
    #        individual = future_ind
    #    else:
    #        pass
    #    population.append(individual)
    #    f.write(individual.to_write)
    #ind_queue = []

    # rank the population using NSG2
    population, front = multiObj(population, k=pop_size, nd='standard')
    global mate_rate
    mate_rate = change_mate_rate(front)

    # create a placeholder variable to trigger off the 'while loop' after the model converges
    not_yet_optimized = True
    gens = 1 #count iterations of generations
    fitn = 0 # keep track of top performing fitness
    while not_yet_optimized:
        gens+=1

        # send population to generate new offspring
        ind_queue, f = generate(population, ind_queue, gens, global_inputs, f)
        try:
            scoop._control.execQueue.updateQueue() # I think this goes here???...taken from jason's code
        except:
            pass

        # while loop to add individuals to population until we have enough to start new generation
        #more_sleep = False
        while len(ind_queue) >= min_queue_size:

            #if more_sleep:
            #    time.sleep(10)
            #else:
            #    pass

            i = 0
            while i < len(ind_queue):
                ind_future = ind_queue[i]
                if compute == "scoop":
                    if ind_future.done():
                        offspring = ind_future.result()
                        population.append(offspring)
                        f.write(offspring.to_write)
                        ind_queue.pop(i)
                    else:
                        i+=1
                elif compute == "local":
                    population.append(ind_future)
                    f.write(str(ind_future.to_write))
                    ind_queue.pop(i)
            # print("\n\n\n\ncurrent length of individual queue:", i, len(ind_queue), flush=True)
            if compute == "scoop":
                ram_file = "/tmp/rt_checkInds_%s-%s-%s" % (i, len(ind_queue), min_queue_size)
                open(ram_file, 'a').close()
            if len(ind_queue) >= min_queue_size and compute=="scoop":
                #more_sleep = True
                time.sleep(60)
            else:
                pass
        try:
            scoop._control.execQueue.updateQueue()
        except:
            pass


        # out of 'while' loop so we are ready to start a new generation...prep first
        population, front = multiObj(population, k=pop_size, nd='standard')
        mate_rate = change_mate_rate(front)

        # create the object of HyperVolume type
        hyperVolume = HyperVolume(referencePoint)

        # collect all fitnesses
        pop_fitness = []
        for indiv in population:
            #temp = list(indiv.fitness.values)
            #for fval, fit in enumerate(temp):
            #    if math.isnan(fit):
            #        temp[fval] = referencePoint[fval]
            #    else:
            #        pass
            #indiv.fitness.values = tuple(temp)
            pop_fitness.append(indiv.fitness.values)
        # calc the hypervolume with all the fitnesses
        #print("\n2fitness...\n", pop_fitness, flush=True)
        volume = hyperVolume.compute(pop_fitness)
        #print("2volume:\n", volume, flush=True)
        if math.isnan(volume):
            volume = 0
        else:
            pass


        ### save fronts to file...store to gen# and store volume of front
        # https://stackoverflow.com/questions/20716812/saving-and-loading-multiple-objects-in-pickle-file
        front_file = "outputs/PFront_run%s_gen%s_vol%s.txt" % (run, gens, int(volume*100))
        print("pickling")
        with open(front_file, 'wb') as p: #p = file(front_file, 'wb')
            for i,indv in enumerate(front):
                pickle.dump(indv, p)
                print("done", i)
        #p.close()
        print("pickling done")
        del front

        to_write = my_logging("generation done", [gens, volume, population[0].fitness.values, population[1].fitness.values,population[2].fitness.values,population[3].fitness.values])
        f.write(to_write)

        # calculate the max possible hypervolume from reference point
        maxVolume = 1
        for i in referencePoint:
            maxVolume *= i
        # check for convergence
        if volume >= maxVolume-epsilon:
            not_yet_optimized = False
            to_write = my_logging("converged", [gens, volume])
            f.write(to_write)
        elif gens >= max_gens:
            not_yet_optimized = False
            to_write = my_logging("converged", [gens, volume])
            f.write(to_write)
        else:
            pass
    f.close()
    # for now return most recent population
    return population