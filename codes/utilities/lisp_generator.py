'''
Playing around to see if I can make a script to facilitate the process of generating a lisp
genome that can be used to start a fresh evolution...say I have a man-made solution and I
want to evolve off of that instead of starting from scratch or seeding from some previous
evolution run.

We already have a get_lisp() method associated with the Block_Definition() class, so maybe
we can steal that method
'''

### packages
import os
import shutil
import numpy as np
from copy import deepcopy
import pdb

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from codes.block_definitions.block_definition import BlockDefinition



class FakeDefinition():
    def __init__(self, input_count, main_count, output_count=1):
        self.input_count = input_count
        self.main_count = main_count
        self.output_count = output_count
        self.genome_count = input_count + main_count + output_count

    def get_actives(self, material):
        BlockDefinition.get_actives(self, material)

    def get_lisp(self, material):
        BlockDefinition.get_lisp(self, material)



class FakeMaterial():
    def __init__(self, genome, args, _id="TEST"):
        self.genome = genome
        self.args = args
        self.lisp = [] # "" ?
        self.active_nodes = []
        self.id = _id

    def __getitem__(self, index):
        return self.genome[index]


def generate_block_seed(genome, args, input_count, main_count, output_count, block_name, save_to=None):
    try:
        material = FakeMaterial(genome, args, "test")
        definition = FakeDefinition(input_count, main_count, output_count)
        definition.get_lisp(material)
    except Exception as err:
        print("oops...something we wrong:\n%s\n" % err)
        pdb.set_trace()
        exit()

    # if we got this far then everything worked correctly.
    # make a folder dir and save the lisp in a text file
    if save_to is None:
        save_to = os.getcwd()

    assert(os.path.exists(save_to)), "wtf mate...how does %s not exist?" % save_to

    with open(os.path.join(save_to, "%s_lisp.txt" % block_name), 'w') as f:
        for lisp in material.lisp:
            f.write("%s\n" % lisp)


def generate_individual_seed(list_of_info, individual_name):
    '''
    kinda lazy and just gonna assume that the user can put together a list of lists where
    the greater list has len == number of blocks and each element is the list of args
    needed for generate_block_seed()
    '''
    save_to = os.path.join(os.getcwd(), "IndivSeed_%s" % individual_name)
    if os.path.exists(save_to):
        response = input("folder exists; are you sure you want to overwrite it? (y/n)... ")
        if response.lower() == "y":
            shutil.rmtree(save_to)
        else:
            print("okay...gonna exit then.")
            exit()

    os.makedirs(save_to)
    for args in list_of_info:
        generate_block_seed(*args, save_to=save_to)



def build_block_from_lisp(lisp):
    '''
    copied the first chunk of this exact method from factory just to verify this statement:
        "we also can handle cases where we are 'reusing' the output of a node"
    '''
    import re

    _active_dict = {}
    ith_node = 0
    while True:
        # from the start of the string, keep looking for lists []
        match = re.search("\[[0-9A-Za-z_\-\s.,']+\]", lisp)
        if match is None:
            # no more lists inside lisp. so we're done
            break
        else:
            # get the single element lisp
            _active_dict[ith_node] = lisp[match.start(): match.end()]
            # now replace that element with the node number
            # add 'n' to distinguish from arg value
            lisp = lisp.replace(_active_dict[ith_node], "%in" % ith_node)
            # increment to next node
            ith_node +=1

            if ith_node >= 10**3:
                # very unlikely to have more than 1000 nodes...prob something went wrong
                print("something went wrong")
                break

    return _active_dict



if __name__ == "__main__":
    '''
    test out the code baby!
    '''
    def my_add(a,b):
        return a+b

    def my_subtract(a,b):
        return a-b

    def my_multiply(a,b):
        return a*b

    # set params
    input_count = 2
    main_count = 100
    output_count = 1
    genome_count = input_count + main_count + output_count
    arg_count = 50

    # define args
    args = list(np.random.randint(100, 200, arg_count))

    # define genome
    #...just arbitrarily fill in args; get_lisp() doesn't use operator dict
    #   so args can be meaningless in context of function
    # BUT loading in from lisp to BlockMaterial DOES(!) use operator dict
    genome = [None]*genome_count
    genome[0] = {"ftn": my_add,
                 "inputs": [-1, -2],
                 "args": [11, 3]}
    genome[1] = {"ftn": my_subtract,
                 "inputs": [0, -2],
                 "args": [2, 43]}
    genome[2] = {"ftn": my_add,
                 "inputs": [-1, 0],
                 "args": [6, 4]}
    genome[3] = {"ftn": my_multiply,
                 "inputs": [2, 1],
                 "args": [25, 33, 39]}
    genome[19] = {"ftn": my_add,
                 "inputs": [2, 3],
                 "args": [19]}
    genome[main_count] = 19


    material = FakeMaterial(genome, args, "poop")
    definition = FakeDefinition(input_count, main_count, output_count)
    definition.get_lisp(material)
    print(material.lisp) # <- should be a list of strings with len == output_count

    deconstructed_lisp = build_block_from_lisp(deepcopy(material.lisp[0]))
    for index, value in deconstructed_lisp.items():
        print("%i: %s" % (index, value))

    generate_individual_seed(list_of_info=[[genome, args, input_count, main_count, output_count, "SOME_BLOCK"]],
                             individual_name="TEST_ARCHITECTURE")