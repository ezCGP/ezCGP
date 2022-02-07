'''
root/code/individual/individual_block.py

Overview:
So similar how a Block has 6 definitions/characteristics, so does an Individual...well, kinda.
An Individual is really just a list of blocks stored in block_defs, and then 3 wrapper methods for mutate, mate, and evaluate.
The 3 individual definitions/characteristics define exactly how to pick which blocks to mutate, mate, or evaluate...that's it.
So while we allow the option for these definitions to get switched out easily, they will likely never change once our experiments
reveal the best way to evolve.
'''

### packages
from typing import List

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from codes.block_definitions.block_definition import BlockDefinition
from codes.individual_definitions.individual_evaluate import IndividualEvaluate_Abstract
from codes.individual_definitions.individual_mutate import IndividualMutate_Abstract
from codes.individual_definitions.individual_mate import IndividualMate_Abstract
from codes.genetic_material import IndividualMaterial, BlockMaterial
from data.data_tools.ezData import ezData
from codes.utilities.custom_logging import ezLogging



class IndividualDefinition():
    '''
    Combine the list of BlockDefinitions, definitions for evaluating, mating, and mutating, together into a class instance...
    Note we pass in the class as arguments to __init__, and then inside the method we instantiate the classes to attributes.
    '''
    def __init__(self,
                 block_defs: List[BlockDefinition],
                 evaluate_def: IndividualEvaluate_Abstract,
                 mutate_def: IndividualMutate_Abstract,
                 mate_def: IndividualMate_Abstract):
        ezLogging.debug("%s-%s - Starting Initialize Individual" % (None, None))
        self.block_defs = block_defs
        self.block_count = len(block_defs)
        self.mutate_def = mutate_def()
        self.mate_def = mate_def()
        self.evaluate_def = evaluate_def()


    def __getitem__(self, block_index: int):
        '''
        just a shortcut way to grab the ith BlockDefinition...self[ith]
        '''
        return self.block_defs[block_index]


    def get_actives(self, indiv_material: IndividualMaterial):
        '''
        loop over each block and set the actives attribute to prep for evaluation
        '''
        ezLogging.debug("%s - Inside get_actives" % (indiv_material.id))
        for block_index in range(self.block_count):
            self[block_index].get_actives(indiv_material[block_index])
    
    
    def postprocess_evolved_individual(self, evolved_material: IndividualMaterial, evolved_block_index: int):
        '''
        An 'evolved' individual is a mutant from mutate() more child from mate()
        
        Since it is a new individual, it needs a new id, and we need to process
        the genome to get the new active nodes and to set to be re-evaluated.
        '''
        evolved_material.set_id()
        ezLogging.debug("%s - Inside postprocess_evolved_individual, block_index: %i, to process a new individual" % (evolved_material.id, evolved_block_index))
        for block_index, block_material in enumerate(evolved_material.blocks):
            if block_index >= evolved_block_index:
                block_material.need_evaluate = True
                block_material.dead = False
                block_material.output = []
        self[evolved_block_index].get_actives(evolved_material[evolved_block_index])
        evolved_material.dead = False
        evolved_material.output = []
        evolved_material.fitness.values = ()


    def mutate(self, indiv_material: IndividualMaterial):
        '''
        wrapper method that just directs mutate call to the IndividualMutate class definition of mutate
        '''
        ezLogging.info("%s - Sending to Individual Mutate Definition" % (indiv_material.id))
        mutants = self.mutate_def.mutate(indiv_material, self)
        return mutants


    def mate(self, parent1: IndividualMaterial, parent2: IndividualMaterial):
        '''
        wrapper method that just directs mate call to the IndividualMate class definition of mate
        '''
        ezLogging.info("%s+%s - Sending to Individuals Mate Definition" % (parent1.id, parent2.id))
        children = self.mate_def.mate(parent1, parent2, self)
        return children


    def evaluate(self, indiv_material: IndividualMaterial, training_datapair: ezData, validation_datapair=None):
        '''
        wrapper method that just directs evaluate call to the IndividualEvaluate class definition of mate
        '''
        ezLogging.warning("%s - Sending to Individual Evaluate Definition" % (indiv_material.id))
        self.evaluate_def.evaluate(indiv_material, self, training_datapair, validation_datapair)