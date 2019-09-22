from utils.skeleton_block import SkeletonBlock
import operators
import numpy as np

class PreprocessingBlock(SkeletonBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "input_dtypes" not in kwargs:
            self.block_input_dtypes = [np.ndarray]
        if "output_dtypes" not in kwargs:
            self.block_outputs_dtypes = [np.ndarray]
        if "setup_dict_ftn" not in kwargs:
            # default layers

            #declare which primitives are available to the genome,
            #and assign a 'prob' so that you can control how likely a primitive will be used;
            #prob: float btwn 0 and 1 -> assigns that prob to that primitive...the sum can't be more than 1
            #prob: 1 -> equally distribute the remaining probability amoung all those remaining (hard to explain, sorry)
            self.setup_dict_ftn = {
                operators.identity_layer: {"prob": 1}
            #     operators.gassuian_blur: {'prob': 1},
            }
        else:
            # specific layers
            pass
