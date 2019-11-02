### individual.py

# external packages
import numpy as np
from copy import deepcopy
from mpi4py import MPI
import logging

# my scripts
from blocks import Block

class Individual(): # Block not inherited in...rather just instanciate to an attribute or something
    """
    Individual genome composed of blocks of smaller genome.
    Here we define those blocks into the full genome
    """
    def __init__(self, skeleton):
        # TODO eventually make it so we can read in an xml/json/yaml file
        # the user will be able to outline the individual.genome there
        # instead of having to edit this file every time

        #self.genome = [] #maybe a dict instead?
        """
        self.skeleton = {
            'input': [datatype, datatype],
            'output': [datatype],
            1: {**kwargs},
            2: {},
            3: {}
        }


        [
        block1 = {"inputs"=[inputs0,inputs1],"args","outputs"},
        block2 = {"inputs"=block1.outputs,"args","outputs"},
        output0 = block2.output[0],
        output1 = block2.output[1],
        inputs1,
        inputs0
        ]
        """
        self.skeleton = deepcopy(skeleton) # needs deepcopy or else it will add 'block_object' to the dictionary
        self.blocks = list(self.skeleton.keys())
        self.blocks.remove('input')
        self.blocks.remove('output')
        self.num_blocks = len(self.blocks)
        # verify that the number of blocks matches the dictionary
        for i in range(1,self.num_blocks+1):
            if i not in skeleton.keys():
                print("UserError: skeleton keys do not have a consistent count of blocks")
                exit()
            else:
                # now build out the block if it exists
                self.skeleton[i]["block_object"] = Block(**self.skeleton[i])

        self.fitness = self.Fitness()
        """
        self.preprocessing = Block(
                                nickname="preprocessing",
                                ftn_dict={"ftnName": {"prob"1:}},
                                arg_dict={"argumentDataType": {"prob":1}},
                                mate_dict={"mateMethod": {"prob":1, "args":[]}},
                                mut_dict={"mutateMethod": {"prob":1, "args":[]}},
                                gene_dict={ftnName: {"inputs":[], "args":[], "output":_}},
                                block_inputs=[inputDataType, inputDataType],
                                block_outputs=[outputDataType, outputDataType],
                                block_main_count=num_main,
                                block_arg_count=num_args)
        """


    def get_genome_list(self):
        genome_list = []
        for i in range(1,self.num_blocks+1):
            print('block number ', i)

            # evaluate the block
            block = self.skeleton[i]["block_object"]
            genome_list.append(block.genome)
        genome_list.append(self.fitness.values)
        return genome_list


    def __getitem__(self, block_index):
        if (block_index=="input") or (block_index=="output"):
            return self.skeleton[block_index]
        else:
            return self.skeleton[block_index]["block_object"]


    def need_evaluate(self):
        for i in range(1,self.num_blocks+1):
            if self.skeleton[i]["block_object"].need_evaluate:
                return True
            else:
                pass
        return False

    def clear_rec(self):
        for i in range(1,self.num_blocks+1):
            block = self.skeleton[i]["block_object"]
            block.rec_clear()




    def evaluate(self, data, labels=None, validation_pair=None):
        # test = 0
        # initcomm = MPI.Comm.Get_parent()
        # initcomm.Bcast(test, root=0)

       # print("LOOK HERE", test)
        # added validation pair support external validation of labels/data in each block
        for i in range(1,self.num_blocks+1):
            print('block number ', i)

            # evaluate the block
            block = self.skeleton[i]["block_object"]
            block.evaluate(block_inputs=data, labels_all=labels, validation_pair=validation_pair)
            data = self.skeleton[i]["block_object"].genome_output_values
            self.skeleton[i]["block_object"].genome_output_values = []
            if block.apply_to_val:
                validation_pair = self.skeleton[i]["block_object"].validation_pair_output
                self.skeleton[i]["block_object"].validation_pair_output = []
            # after running enumerate in individual.evaluate(), data strips down one dimension, so add back the dimension
            if block.tensorblock_flag and i != self.num_blocks:
                new_shape = (1,) + data.shape
                data = data.reshape(new_shape)
        self.genome_outputs = data
        #return genome_outputs

    def score_fitness(self, labels):
        #self.fitness =
        pass


    def mutate(self, block=None):
        if block is "random selection":
            roll = np.random.random()
            for i in range(1,self.num_blocks):
                if roll <= i/self.num_blocks:
                    self.skeleton[i]["block_object"].mutate()
                    break
                else:
                    continue
        elif block is None:
            for i in range(self.num_blocks):
                self.skeleton[i+1]['block_object'].mutate()
        else:
            self.skeleton[block]["block_object"].mutate()


    def mate(self, other, block=None):
        if block is "random selection":
            roll = np.random.random()
            for i in range(1,self.num_blocks):
                if roll <= i/self.num_blocks:
                    offspring_list = self.skeleton[i]["block_object"].mate(
                                            other.skeleton[i]["block_object"])
                    break
                else:
                    continue
        elif block is None:
            for i in range(1,self.num_blocks):
                offspring_list = self.skeleton[i]["block_object"].mate(
                                        other.skeleton[i]["block_object"])
        else:
            offspring_list = self.skeleton[i]["block_object"].mate(
                                    other.skeleton[i]["block_object"])
        return offspring_list


    class Fitness(object):
        '''
        the NSGA taken from deap requires a Fitness class to hold the values.
        so this attempts to recreate the bare minimums of that so that NSGA
        or (hopefully) any other deap mutli obj ftn handles this Individual class
        http://deap.readthedocs.io/en/master/api/base.html#fitness
        '''

        def __init__(self):
            self.values = () #empty tuple

        # check dominates
        def dominates(self, other):
            a = np.array(self.values)
            b = np.array(other.values)
            # 'self' must be at least as good as 'other' for all objective fnts (np.all(a>=b))
            # and strictly better in at least one (np.any(a>b))
            return np.any(a < b) and np.all(a <= b)


def create_individual_from_genome_list(skeleton_genome, genome_list):
    individual = Individual(skeleton_genome)
    for i in range(1,individual.num_blocks+1):
        # replace with genome
        individual.skeleton[i]["block_object"].genome = genome_list[i-1]

    return individual
