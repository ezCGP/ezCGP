'''
use utilities/lisp_generator.py to seed simgan refiner and discriminator blocks
'''

### packages
import numpy as np
import torch

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from codes.utilities import lisp_generator
import codes.block_definitions.utilities.operators_pytorch as opPytorch
import codes.block_definitions.utilities.operators_simgan_train_config as opTrainConfig
from codes.block_definitions.shapemeta.block_shapemeta import (BlockShapeMeta_SimGAN_Network,
                                                               BlockShapeMeta_SimGAN_Train_Config,
                                                               BlockShapeMeta_SimGAN_Train_Config)
from codes.block_definitions.evaluate.block_evaluate_pytorch import (BlockEvaluate_SimGAN_Refiner,
                                                                     BlockEvaluate_SimGAN_Discriminator,
                                                                     BlockEvaluate_SimGAN_Train_Config)

# to be used later for writing the seed
block_seed_info = []

### REFINER
# CONSTANTS
in_channels = 5
nb_channels = 10
kernel = 5
padding = 2
use_leaky_relu = True
num_blocks = 2

shapemeta_refiner = BlockShapeMeta_SimGAN_Network()
args = []
genome = [None]*shapemeta_refiner.genome_count

this_args = [nb_channels, kernel, 1, padding, None]
genome[0] = {"ftn": opPytorch.conv1d_layer,
             "inputs": [-1],
             "args": list(np.arange(len(this_args))+len(args))}
args+=this_args

for i in range(num_blocks):
    this_args = [nb_channels, kernel, use_leaky_relu]
    genome[i+1] = {"ftn": opPytorch.resnet,
                   "inputs": [i],
                   "args": list(np.arange(len(this_args))+len(args))}
    args+=this_args

genome[shapemeta_refiner.main_count] = i+1

material = lisp_generator.FakeMaterial(genome, args, "poop")
definition = lisp_generator.FakeDefinition(shapemeta_refiner.input_count, shapemeta_refiner.main_count, shapemeta_refiner.output_count)
definition.get_lisp(material)
block_seed_info.append([genome,
                        args,
                        shapemeta_refiner.input_count,
                        shapemeta_refiner.main_count,
                        shapemeta_refiner.output_count,
                        "RefinerBlock"])

''' Verify lisp worked...
print(material.lisp)
# now try evaluating to see that evaluate_def works and graph builds
evaluate_def = BlockEvaluate_SimGAN_Refiner()
fake_data = torch.randn(500,1,92)
evaluate_def.evaluate(material, definition, [fake_data], None, [])
print("Built graph!")
output = material.graph(fake_data)
print("\n...go to discrim\n")
'''



### DISCRIM
# CONSTANTS
pixel_length = 6
mbd_kernel_dims = 5

shapemeta_discrim = BlockShapeMeta_SimGAN_Network()
args = []
genome = [None]*shapemeta_discrim.genome_count

this_args = [64, 5, 2, 2, torch.nn.LeakyReLU(0.1)]
genome[0] = {"ftn": opPytorch.conv1d_layer,
             "inputs": [-1],
             "args": list(np.arange(len(this_args))+len(args))}
args+=this_args

this_args = [32, 5, 2, 2, torch.nn.LeakyReLU(0.1)]
genome[1] = {"ftn": opPytorch.conv1d_layer,
             "inputs": [0],
             "args": list(np.arange(len(this_args))+len(args))}
args+=this_args

this_args = []
genome[2] = {"ftn": opPytorch.batch_normalization,
             "inputs": [1],
             "args": list(np.arange(len(this_args))+len(args))}
args+=this_args

this_args = [3, 1, 1]
genome[3] = {"ftn": opPytorch.avg_pool,
             "inputs": [2],
             "args": list(np.arange(len(this_args))+len(args))}
args+=this_args

this_args = [16, 1, 2, 0, torch.nn.LeakyReLU(0.1)]
genome[4] = {"ftn": opPytorch.conv1d_layer,
             "inputs": [3],
             "args": list(np.arange(len(this_args))+len(args))}
args+=this_args

this_args = []
genome[5] = {"ftn": opPytorch.batch_normalization,
             "inputs": [4],
             "args": list(np.arange(len(this_args))+len(args))}
args+=this_args

this_args = [8, 1, 2, 0, torch.nn.LeakyReLU(0.1)]
genome[6] = {"ftn": opPytorch.conv1d_layer,
             "inputs": [5],
             "args": list(np.arange(len(this_args))+len(args))}
args+=this_args

this_args = []
genome[7] = {"ftn": opPytorch.flatten_layer,
             "inputs": [6],
             "args": list(np.arange(len(this_args))+len(args))}
args+=this_args

this_args = [4*pixel_length, mbd_kernel_dims]
genome[8] = {"ftn": opPytorch.minibatch_discrimination,
             "inputs": [7],
             "args": list(np.arange(len(this_args))+len(args))}
args+=this_args

this_args = []
genome[9] = {"ftn": opPytorch.feature_extraction,
             "inputs": [-1],
             "args": list(np.arange(len(this_args))+len(args))}
args+=this_args

this_args = [1]
genome[10] = {"ftn": opPytorch.pytorch_concat,
             "inputs": [7, 8],
             "args": list(np.arange(len(this_args))+len(args))}
args+=this_args

this_args = [1]
genome[11] = {"ftn": opPytorch.pytorch_concat,
             "inputs": [10, 9],
             "args": list(np.arange(len(this_args))+len(args))}
args+=this_args

this_args = [0.5]
genome[12] = {"ftn": opPytorch.dropout,
             "inputs": [11],
             "args": list(np.arange(len(this_args))+len(args))}
args+=this_args

genome[shapemeta_discrim.main_count] = 12

material = lisp_generator.FakeMaterial(genome, args, "poop")
definition = lisp_generator.FakeDefinition(shapemeta_discrim.input_count, shapemeta_discrim.main_count, shapemeta_discrim.output_count)
definition.get_lisp(material)
block_seed_info.append([genome,
                        args,
                        shapemeta_discrim.input_count,
                        shapemeta_discrim.main_count,
                        shapemeta_discrim.output_count,
                        "DiscriminatorBlock"])


'''# Verify that the lisp works...
print(material.lisp)
# now try evaluating to see that evaluate_def works and graph builds
evaluate_def = BlockEvaluate_SimGAN_Discriminator()
fake_data = torch.randn(500,1,92)
evaluate_def.evaluate(material, definition, [fake_data], None, [])
print("Built graph!")
output = material.graph(fake_data)
'''

### Train Config
shapemeta_trainconfig = BlockShapeMeta_SimGAN_Train_Config()
args = []
genome = [None]*shapemeta_trainconfig.genome_count

this_args = [5000, 500, 400, 1, 2, 0.001, 0.0001, 0.0001, True, 0, True, 4]
genome[0] = {"ftn": opTrainConfig.simgan_train_config,
             "inputs": [-1],
             "args": list(np.arange(len(this_args))+len(args))}
args+=this_args

genome[shapemeta_trainconfig.main_count] = 0

material = lisp_generator.FakeMaterial(genome, args, "poop")
definition = lisp_generator.FakeDefinition(shapemeta_trainconfig.input_count, shapemeta_trainconfig.main_count, shapemeta_trainconfig.output_count)
material.evaluated = [None]*definition.genome_count
material.dead = False
definition.input_dtypes = [dict]
definition.get_lisp(material)
block_seed_info.append([genome,
                        args,
                        shapemeta_trainconfig.input_count,
                        shapemeta_trainconfig.main_count,
                        shapemeta_trainconfig.output_count,
                        "ConfigBlock"])

''' # Verify lisp working
print(material.lisp)
# now try evaluating to see that evaluate_def works and graph builds
evaluate_def = BlockEvaluate_SimGAN_Train_Config()
fake_data = torch.randn(500,1,92)
evaluate_def.evaluate(material, definition, [fake_data], None, [])
print("Built graph!")
config = material.output[-1][0] # get from supplements
'''



# COMPILE AND SAVE!
lisp_generator.generate_individual_seed(list_of_info=block_seed_info,
                                        individual_name="SimGAN_Seed0")
