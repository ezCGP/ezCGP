'''
root/code/factory.py

Overview:
factory class will be tasked with building/__init__ all the other classes.
that way we can wrap other debugging + logging items around the init of each class

Rules:
mention any assumptions made in the code or rules about code structure should go here
'''

### packages

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from codes.population import PopulationDefinition
from codes.genetic_material import IndividualMaterial, BlockMaterial
from codes.individual_definitions.individual_definition import IndividualDefinition
from codes.block_definitions.block_definition import BlockDefinition



class FactoryDefinition():
    '''
    TODO
    '''
    def __init__(self):
        pass


    def build_population(self, indiv_def: IndividualDefinition, population_size: int):
        '''
        TODO
        '''
        my_population = PopulationDefinition(population_size)
        for i in range(population_size):
            indiv = self.build_individual(indiv_def, indiv_id="initPop%i" % i)
            my_population.population.append(indiv)
        return my_population


    '''
    def build_subpopulation(self, indiv_def: IndividualDefinition, population_size, number_subpopulation=None, subpopulation_size=None):
        ''
        TODO
        ''
        my_population = SubPopulationDefinition(population_size, number_subpopulation, subpopulation_size)
        for ith_subpop, subpop_size in my_population.subpop_size:
            for _ in range(subpop_size):
                indiv = self.build_individual(indiv_def)
                my_population[ith_subpop].append(indiv)
        return my_population
    '''


    def build_individual(self, indiv_def: IndividualDefinition, indiv_id=None):
        '''
        TODO
        '''
        indiv_material = IndividualMaterial()
        indiv_material.set_id(indiv_id)
        for block_def in indiv_def.block_defs:
            block_material = self.build_block(block_def, indiv_id=indiv_id)
            indiv_material.blocks.append(block_material)
        return indiv_material


    def build_block(self, block_def: BlockDefinition, indiv_id):
        '''
        TODO
        '''
        block_material = BlockMaterial(block_def.nickname)
        block_material.set_id(indiv_id)
        self.fill_args(block_def, block_material)
        self.fill_genome(block_def, block_material)
        block_def.get_actives(block_material)
        return block_material


    def fill_args(self, block_def: BlockDefinition, block_material: BlockMaterial):
        '''
        TODO
        '''
        block_material.args = [None]*block_def.arg_count
        for arg_index, arg_type in enumerate(block_def.arg_types):
            block_material.args[arg_index] = arg_type()


    def fill_genome(self, block_def: BlockDefinition, block_material: BlockMaterial):
        '''
        TODO
        '''
        block_material.genome = [None]*block_def.genome_count
        block_material.genome[(-1*block_def.input_count):] = ["InputPlaceholder"]*block_def.input_count

        # fill main nodes
        for node_index in range(block_def.main_count):
            ftns = block_def.get_random_ftn(return_all=True)
            for ftn in ftns:
                # find inputs
                input_dtypes = block_def.operator_dict[ftn]["inputs"]
                input_index = [None]*len(input_dtypes)
                for ith_input, input_dtype in enumerate(input_dtypes):
                    input_index[ith_input] = block_def.get_random_input(block_material, req_dtype=input_dtype, _max=node_index)
                if None in input_index:
                    # failed to fill it in; try another ftn
                    continue
                else:
                    pass

                # find args
                arg_dtypes = block_def.operator_dict[ftn]["args"]
                arg_index = [None]*len(arg_dtypes)
                for ith_arg, arg_dtype in enumerate(arg_dtypes):
                    arg_index[ith_arg] = block_def.get_random_arg(req_dtype=arg_dtype)
                if None in arg_index:
                    # failed to fill it in; try another ftn
                    continue
                else:
                    pass

                # all complete
                block_material[node_index] = {"ftn": ftn,
                                    "inputs": input_index,
                                    "args": arg_index}
                break
            # error check that node got filled
            if block_material[node_index] is None:
                print("GENOME ERROR: no primitive was able to fit into current genome arrangment")
                exit()

        # fill output nodes
        for ith_output, node_index in enumerate(range(block_def.main_count, block_def.main_count+block_def.output_count)):
            req_dtype = block_def.output_dtypes[ith_output]
            block_material[node_index] = block_def.get_random_input(block_material, req_dtype=req_dtype, _min=0, _max=block_def.main_count)