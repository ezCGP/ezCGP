'''
root/code/genetic_material.py

Overview:
Script to define the unique representation of each individual
And individual is comprised of 'n' number of blocks. So an individual
will be an instance of IndividualMaterial() which itself is primarily
a list of BlockMaterial() instances.

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages
import numpy as np
from datetime import datetime
import logging

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root



class IndividualMaterial():
    '''
    attributes:
     * blocks: list of BlockMaterial instances
     * fitness: instance of class Fitness which is required for MultiObjective Optimization

    methods:
     * need_evalute: checks the respective boolean flag in all blocks
     and returns True if at least any single block is True
    '''
    def __init__(self, _id=None):
        self.id = "default"
        self.fitness = self.Fitness()
        self.blocks = []
        self.output = []
        self.dead = False


    def __setitem__(self, block_index, block_material):
        '''
        TODO
        '''
        self.blocks[block_index] = block_material


    def __getitem__(self, block_index):
        '''
        TODO
        '''
        return self.blocks[block_index]


    def set_id(self, _id=None):
        '''
        thought there could be a unique way to identify an individual for logging and saving
        '''
        if _id is None:
            new = hex(int(datetime.now().strftime("%H%M%S%f")))[2:]
            logging.debug("New ID %s" % new)
            self.id = new
        else:
            logging.debug("New ID %s; given" % _id)
            self.id = _id

        for block in self.blocks:
            block.set_id(indiv_id=self.id)


    def need_evaluate(self):
        '''
        TODO
        '''
        for block in self.blocks:
            if block.need_evaluate:
                return True
        return False


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



class BlockMaterial():
    '''
    attributes:
     * genome: list of mostly dictionaries
     * args: list of args
     * need_evaluate: boolean flag
     * output: TODO maybe have a place to add the output after it has been evaluated
    '''
    def __init__(self, block_nickname="nickname"):
        '''
        sets these attributes:
         * need_evaluate = False
         * genome
         * args
         * active_nodes
         * active_args

        moved to factory
        '''
        self.block_nickname = block_nickname
        self.id = "default-nickname"
        self.genome = []
        self.active_nodes = []
        self.args = []
        self.active_args = []
        self.need_evaluate = True
        # maybe remove these if they don't get used in non-symbolic regression evaluation problems
        self.evaluated = []
        self.output = []
        self.dead = False


    def __setitem__(self, node_index, value):
        '''
        TODO
        '''
        self.genome[node_index] = value


    def __getitem__(self, node_index):
        '''
        TODO
        '''
        return self.genome[node_index]
    
    def set_id(self, indiv_id):
        self.id = "%s-%s" % (indiv_id, self.block_nickname)