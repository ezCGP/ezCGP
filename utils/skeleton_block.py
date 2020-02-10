import arguments
import mutate_methods as mut
import operators
import tensorflow as tf
import os

# outlines parameters to be passed into a Block object
# this is done in individual.py in the constructer
class SkeletonBlock:

    def __init__(self,
                 tensorblock_flag=True,
                 learning_required=False,
                 apply_to_val = True,
                 batch_size=128,
                 n_epochs=50,
                 large_dataset=None,
                 nickname='',
                 primitives={},
                 args={
                     arguments.argPow2: {'prob': 1},
                     arguments.argFilterSize: {'prob': 1},
                     arguments.argKernelSize: {'prob': 1},
                     arguments.percentage: {'prob': 1},
                     arguments.rotRange: {'prob': 1},
                     arguments.argPoolHeight: {'prob': 1},
                     arguments.argPoolWidth: {'prob': 1},
                     arguments.activation: {"prob": 1}
                 },
                 mut={
                     mut.Mutate.mutate_singleInput: {'prob': 1, 'args': []},
                     mut.Mutate.mutate_singleArgValue: {'prob': 1, 'args': []},
                     mut.Mutate.mutate_singleArgIndex: {'prob': 1, 'args': []},
                     mut.Mutate.mutate_singleFtn: {'prob': 1, 'args': []},
                 },
                 operator_dict=operators.operDict,
                 input_dtypes=[tf.Tensor],
                 output_dtypes=[tf.Tensor],
                 main_count=50, # max number of genes
                 arg_count=20, # max number of arguments per primitive
                 mut_prob=1):

        self.tensorblock_flag = tensorblock_flag
        self.learning_required = learning_required
        self.apply_to_val = apply_to_val
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.large_dataset = large_dataset
        self.nickname = nickname
        self.setup_dict_ftn = primitives
        self.setup_dict_arg = args
        self.setup_dict_mut = mut
        self.operator_dict = operator_dict
        self.block_input_dtypes = input_dtypes
        self.block_outputs_dtypes = output_dtypes
        self.block_main_count = main_count
        self.block_arg_count = arg_count
        self.block_mut_prob = mut_prob
