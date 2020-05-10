'''
root/code/individual/individual_block.py

Overview:
defined by list of blocks

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages

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
from data.data_tools.data_types import ezDataSet



class IndividualDefinition():
    '''
    TODO
    '''
    def __init__(self,
                 block_defs: List[BlockDefinition],
                 evaluate_def: IndividualEvaluate_Abstract,
                 mutate_def: IndividualMutate_Abstract,
                 mate_def: IndividualMate_Abstract):
        self.block_defs = block_defs
        self.block_count = len(block_defs)
        self.mutate_def = mutate_def()
        self.mate_def = mate_def()
        self.evaluate_def = evaluate_def()


    def __getitem__(self, block_index: int):
        '''
        TODO
        '''
        return self.block_defs[block_index]


    def get_actives(self, indiv_material: IndividualMaterial):
        '''
        TODO
        '''
        for block_index, block in enumerate(indiv_material.blocks):
            self[block_index].get_actives(indiv_material[block_index])


    def mutate(self, indiv_material: IndividualMaterial):
        '''
        TODO
        '''
        mutants = self.mutate_def.mutate(indiv_material, self)
        return mutants


    def mate(self, parent1: IndividualMaterial, parent2: IndividualMaterial):
        '''
        TODO
        '''
        children = self.mate_def.mate(parent1, parent2, self)
        return children


    def evaluate(self, indiv_material: IndividualMaterial, training_datapair: ezDataSet, validation_datapair=None):
        '''
        TODO
        '''
        self.evaluate_def.evaluate(self, indiv_material, training_datapair, validation_datapair)