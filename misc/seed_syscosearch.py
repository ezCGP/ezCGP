'''
follow misc/seed_simgan.py to similiarly seed for SyscoSearch

for now, building out a single seed based off the currently used 'experiment_configs_v2.json'
'''

### packages
import numpy as np

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from codes.utilities import lisp_generator
import codes.block_definitions.utilities.operators_SyscoSearch as opSyscoSearch
from codes.block_definitions.shapemeta.block_shapemeta import BlockShapeMeta_Sysco
from codes.block_definitions.evaluate.block_evaluate import BlockEvaluate_MiddleBlock, BlockEvaluate_FinalBlock


def build_individual_seed(synonym_args=['optdefault'],
                          productdesc_args=[5,3,8,2],
                          stocked_args=[25,3,2],
                          addboosts_args=[6,10,2.5],
                          rankeq_args=['base_query.json'],
                          tag=None):
    
    # to be used later for writing the seed
    block_seed_info = []

    ### 1 - Synonym
    shapemeta = BlockShapeMeta_Sysco()
    genome = [None]*shapemeta.genome_count
    genome[0] = {"ftn": opSyscoSearch.pick_synomym_filter,
                "inputs": [-1],
                "args": [0]}
    genome[shapemeta.main_count] = 0
    args = synonym_args
    material = lisp_generator.FakeMaterial(genome, args, "poop")
    definition = lisp_generator.FakeDefinition(shapemeta.input_count, shapemeta.main_count, shapemeta.output_count)
    definition.get_lisp(material)
    block_seed_info.append([genome,
                            args,
                            shapemeta.input_count,
                            shapemeta.main_count,
                            shapemeta.output_count,
                            "synonym_block"])


    ### 2 - product_desc
    shapemeta = BlockShapeMeta_Sysco()
    genome = [None]*shapemeta.genome_count
    genome[0] = {"ftn": opSyscoSearch.pick_productdesc_boosts,
                "inputs": [-1],
                "args": [0,1,2,3]}
    genome[shapemeta.main_count] = 0
    args = productdesc_args
    material = lisp_generator.FakeMaterial(genome, args, "poop")
    definition = lisp_generator.FakeDefinition(shapemeta.input_count, shapemeta.main_count, shapemeta.output_count)
    definition.get_lisp(material)
    block_seed_info.append([genome,
                            args,
                            shapemeta.input_count,
                            shapemeta.main_count,
                            shapemeta.output_count,
                            "product_description_block"])


    ### 3 - stocked
    shapemeta = BlockShapeMeta_Sysco()
    args = []
    genome = [None]*shapemeta.genome_count

    genome[0] = {"ftn": opSyscoSearch.pick_stock_boosts,
                "inputs": [-1],
                "args": [0,1,2]}
    genome[shapemeta.main_count] = 0
    args = stocked_args
    material = lisp_generator.FakeMaterial(genome, args, "poop")
    definition = lisp_generator.FakeDefinition(shapemeta.input_count, shapemeta.main_count, shapemeta.output_count)
    definition.get_lisp(material)
    block_seed_info.append([genome,
                            args,
                            shapemeta.input_count,
                            shapemeta.main_count,
                            shapemeta.output_count,
                            "stocked_flag_block"])


    ### 4 - additional
    shapemeta = BlockShapeMeta_Sysco()
    args = []
    genome = [None]*shapemeta.genome_count
    genome[0] = {"ftn": opSyscoSearch.pick_additional_boosts,
                "inputs": [-1],
                "args": [0,1,2]}
    genome[shapemeta.main_count] = 0
    args = addboosts_args
    material = lisp_generator.FakeMaterial(genome, args, "poop")
    definition = lisp_generator.FakeDefinition(shapemeta.input_count, shapemeta.main_count, shapemeta.output_count)
    definition.get_lisp(material)
    block_seed_info.append([genome,
                            args,
                            shapemeta.input_count,
                            shapemeta.main_count,
                            shapemeta.output_count,
                            "additional_boosts_block"])


    ### 5 - rank_equation
    shapemeta = BlockShapeMeta_Sysco()
    args = []
    genome = [None]*shapemeta.genome_count
    genome[0] = {"ftn": opSyscoSearch.pick_rank_equation,
                "inputs": [-1],
                "args": [0]}
    genome[shapemeta.main_count] = 0
    args = rankeq_args
    material = lisp_generator.FakeMaterial(genome, args, "poop")
    definition = lisp_generator.FakeDefinition(shapemeta.input_count, shapemeta.main_count, shapemeta.output_count)
    definition.get_lisp(material)
    block_seed_info.append([genome,
                            args,
                            shapemeta.input_count,
                            shapemeta.main_count,
                            shapemeta.output_count,
                            "ranking_equation_block"])


    # COMPILE AND SAVE!
    name = "SyscoSearch_Seed"
    if tag is not None:
        name += "_%s" % tag
    seed_folder = lisp_generator.generate_individual_seed(list_of_info=block_seed_info,
                                                          individual_name=name)
    return seed_folder


if __name__=="__main__":
    _ = build_individual_seed(synonym_args=['optdefault'],
                              productdesc_args=[5,3,8,2],
                              stocked_args=[25,3,2],
                              addboosts_args=[6,10,2.5],
                              rankeq_args=['base_query.json'],
                              tag='originalv1')
    _ = build_individual_seed(synonym_args=['optdefault'],
                              productdesc_args=[5,3,8,2],
                              stocked_args=[10,3,2], # <- only difference
                              addboosts_args=[6,10,2.5],
                              rankeq_args=['base_query.json'],
                              tag='originalv2')