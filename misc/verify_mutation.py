'''
Issue 162
Hemang+Daniel were not convinced that the mutation was working correctly because so much of the population had the same looking genome
'''

# external modules
import os
import pickle as pkl
import pdb

# to load in the individual definitions
from problems.problem_cifar_no_transfer import Problem

# for the genome part:
from codes.genetic_material import IndividualMaterial

# load in problem to get individual def
problem = Problem()

# path to the pkl individual you want to load
indiv_file = ""

# load in individual 
with open(indiv_file, "rb") as f:
    individual = pkl.load(f)

# check out individual
for ith_block, block_material in enumerate(individual.blocks):
    print(block_material.genome)
    print(block_material.active_nodes)
    problem.indiv_def[ith_block].get_lisp(block_material)
    print(block_material.lisp)

# mimic how mutation occurs:
#https://github.com/ezCGP/ezCGP/blob/2021S-BaseCodeDevelopment/codes/universe.py#L118
generation = 0
while True:
    mutants = problem.indiv_def.mutate(individual)
    for ith_mutant, mutant in enumerate(mutants):
        print("Investigate mutant %i from generation %i" % (ith_mutant, generation))
        pdb.set_trace()
    generation +=1