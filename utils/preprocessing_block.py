from utils.skeleton_block import SkeletonBlock
import operators

class PreprocessingBlock(SkeletonBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "setup_dict_ftn" not in kwargs:
            # default layers

            #declare which primitives are available to the genome,
            #and assign a 'prob' so that you can control how likely a primitive will be used;
            #prob: float btwn 0 and 1 -> assigns that prob to that primitive...the sum can't be more than 1
            #prob: 1 -> equally distribute the remaining probability amoung all those remaining (hard to explain, sorry)
            self.setup_dict_ftn = {
                operators.gassuian_blur: {'prob': 1},
            }
        else:
            # specific layers
            pass
