import arguments
import mate_methods as mate
import mutate_methods as mut
import operators
import tensorflow as tf


class SkeletonBlock:

    def __init__(self,
                 tensorblock_flag=True,
                 learning_required=False,
                 batch_size=128,
                 n_epochs=1,
                 large_dataset=None,
                 nickname='tensor_mnist_block',
                 primitives={},
                 args={
                     arguments.argPow2: {'prob': 1},
                     arguments.argFilterSize: {'prob': 1}
                 },
                 mate={mate.Mate.dont_mate: {'prob': 1, 'args': []}},
                 mut={
                     mut.Mutate.mutate_singleInput: {'prob': 1, 'args': []},
                     mut.Mutate.mutate_singleArgValue: {'prob': 1, 'args': []},
                     mut.Mutate.mutate_singleArgIndex: {'prob': 1, 'args': []},
                     mut.Mutate.mutate_singleFtn: {'prob': 1, 'args': []},
                 },
                 operator_dict=operators.operDict,
                 input_dtypes=[tf.Tensor],
                 output_dtypes=[tf.Tensor],
                 main_count=4,
                 arg_count=20,
                 mut_prob=1,
                 mate_prob=0
                 ):
        self.tensorblock_flag = tensorblock_flag
        self.learning_required = learning_required
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.large_dataset = large_dataset
        self.nickname = nickname
        self.setup_dict_ftn = primitives
        self.setup_dict_arg = args
        self.setup_dict_mate = mate
        self.setup_dict_mut = mut
        self.operator_dict = operator_dict
        self.block_input_dtypes = input_dtypes
        self.block_outputs_dtypes = output_dtypes
        self.block_main_count = main_count
        self.block_arg_count = arg_count
        self.block_mut_prob = mut_prob
        self.block_mate_prob = mate_prob


