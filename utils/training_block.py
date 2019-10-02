from utils import SkeletonBlock
import operators

class TrainingBlock(SkeletonBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "apply_to_val" not in kwargs:
            self.apply_to_val = False

        if "setup_dict_ftn" not in kwargs:
            # default layers

            #declare which primitives are available to the genome,
            #and assign a 'prob' so that you can control how likely a primitive will be used;
            #prob: float btwn 0 and 1 -> assigns that prob to that primitive...the sum can't be more than 1
            #prob: 1 -> equally distribute the remaining probability amoung all those remaining (hard to explain, sorry)
            self.setup_dict_ftn = {
                operators.dense_layer: {'prob': 1},
                # operators.input_layer: {'prob': 1},
                # operators.conv_layer: {'prob': 1},
                # operators.max_pool_layer: {'prob': 1},
                # operators.avg_pool_layer: {'prob': 1},
                # operators.concat_func: {'prob': 1},
                # operators.sum_func: {'prob': 1},
                # operators.conv_block: {'prob': 1},
                # operators.res_block: {'prob': 1},
                # operators.sqeeze_excitation_block: {'prob': 1},
                # operators.input_layer: {'prob': 1},	
                #operators.conv_layer: {'prob': 1},	
                # operators.max_pool_layer: {'prob': 1},	
                # operators.avg_pool_layer: {'prob': 1},	
                # operators.concat_func: {'prob': 1},	
                # operators.sum_func: {'prob': 1},	
                # operators.conv_block: {'prob': 1},	
                # operators.res_block: {'prob': 1},	
                # operators.sqeeze_excitation_block: {'prob': 1},	
                # operators.identity_block: {'prob': 1}, # TODO replace this with info from operator_dict?
            }
        else:
            # specific layers
            pass
