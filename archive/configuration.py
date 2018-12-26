### configuration.py



# run variables
runs = 1
pop_size = 2**3
seed = 111096#1313133
verbose = True
streamData = False #no stream data...all feature
min_queue_size = 2**4
compute = "local" # or "scoop"
check_types = True #are we testing, do we want to checkTypes
#memory_limit_perc = 30# SHOULD BE A PERCENT...NOT..7680
#memory_limit_mb = 4096
memory_limit = [30, 4096] # [% , MB]
time_limit = 1800


# node counts:
numInputs = 1
numMain = 100
numOutputs = 1
set_numArgs = 200
perc_args = .5
learner_bias = 0.2

# mutation
ind_mut_rate = 1#.5
mut_rate = .05
ftn_mut_rate = .0
input_mut_rate = .0
argIndex_mut_rate = .0
argValue_mut_rate = 1
duplicate = "accumulate"#"single"
mutant_count = 4

# mating
adaptable = False
set_mate_rate = .5
mate_type = "subgraph"
arg_mate_type = 'single point'
keep_both_offspring = True

# other evolution
ordering = "none" #"reorder"

# converging
max_gens = 10
referencePoint = [1.,1.,2.]
epsilon = .25