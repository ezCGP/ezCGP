'''
root/misc/file.py
we want to do more things with a trained individual
so we need an easy way to load in models and use them with pytorch
'''

### packages
import os
from copy import deepcopy
import torch
import numpy as np

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
PROBLEM_TYPE = "ecg"
if PROBLEM_TYPE == "":
    from problems.problem_simgan import Problem
elif PROBLEM_TYPE == "ecg":
    from problems.problem_simgan_ecg import Problem
elif PROBLEM_TYPE == "transform":
    from problems.problem_simgan_transform import Problem
else:
    print("don't have that one yet")
    exit()



def build_individual(individual_pkl):
    '''
    a lot of this is just going to mimic factory.Factory_SimGAN.build_individual_from_seed
    but use the trained weights instead!
    '''
    problem = Problem()
    indiv_def = problem.indiv_def
    factory = problem.Factory()

    # try to grab id from name
    try:
        indiv_id = os.path.basename(individual_pkl).split("_")[-1].split(".")[0]
    except Exception as err:
        print("Couldn't get id for some reason:\n%s" % err)
        indiv_id = "poopie_face"

    individual = factory.build_individual_from_seed(indiv_def,
                                                    individual_pkl,
                                                    problem.maximize_objectives,
                                                    indiv_id)

    refiner, discriminator = deepcopy(individual.output)
    del individual

    return refiner, discriminator


if __name__ == "__main__":
    '''
    python misc/load_trained_simgan.py -p test_area/univ0000/gen_0008_indiv_2d311ddb156f0.pkl
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pkl",
                        type = str,
                        required = True,
                        help = "full path to individual pkl file")
    args = parser.parse_args()
    refiner, discriminator = build_individual(args.pkl)


    # Test with some fake data
    if PROBLEM_TYPE == "ecg":
        fake_data = torch.randn((999,1,3600))
    else:
        fake_data = torch.randn((999,1,92))

    fake_refine_output = refiner(fake_data)
    fake_discrim_output = discriminator(fake_data)