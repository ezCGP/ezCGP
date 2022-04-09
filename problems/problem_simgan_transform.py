### packages


### sys relative to root dir
import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems import problem_simgan
from data.data_tools import simganData


class Problem(problem_simgan.Problem):
    """
    Basically the same as the other simgan problem but we want to use a different dataset.
    This allows us to toggle between them really easily.
    """
    def __init__(self):
        super().__init__()


    def construct_dataset(self):
        """
        Constructs a train and validation 1D signal datasets
        """
        # Can configure the real and simulated sizes + batch size, but we will use default
        train_config_dict = {"device": "cuda"}  # was gpu but that didn't work anymore
        self.training_datalist = [simganData.TransformSimGANDataset(real_size=512, sim_size=128**2, batch_size=128),
                                  train_config_dict]
        self.validating_datalist = [simganData.TransformSimGANDataset(real_size=128, sim_size=int((128**2)/4), batch_size=128)]