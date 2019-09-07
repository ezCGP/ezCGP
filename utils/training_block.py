from utils import SkeletonBlock
import operators


class TrainingBlock(SkeletonBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "setup_dict_ftn" not in kwargs :
            self.setup_dict_ftn = {operators.dense_layer: {'prob': 1}}
