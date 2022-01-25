'''
root/codes/block_definitions/utilities/operators...

Overview:
This file has operators for a SimGAN training config

Rules:
'''
### packages
import torch

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.block_definitions.utilities import argument_types
from codes.utilities.custom_logging import ezLogging

### init dict
operator_dict = {}

# NOTE: this function needs a placeholder because you aren't allowed to have no inputs, so a None is passed in as the first arg
def simgan_train_config(config,
                        train_steps,
                        r_pretrain_steps,
                        d_pretrain_steps,
                        d_updates_per_train_step,
                        r_updates_per_train_step,
                        r_lr,
                        d_lr,
                        delta,
                        use_data_history,
                        steps_per_log=100):
    """
    An operator that stores SimGAN training configuration variables
    """
    # TODO: move device somewhere else where it can be controlled based on gpu/cpu, maybe into the problem file
    if 'device' not in config:
        config['device'] = torch.device('cpu')
        ezLogging.debug("'device' not set elsewhere so defaulting to 'cpu'.")

    # TODO: Consider finding a way to measure convergence vs. making hardcoding training steps
    config['train_steps'] = train_steps
    config['r_pretrain_steps'] = r_pretrain_steps
    config['d_pretrain_steps'] = d_pretrain_steps
    config['d_updates_per_train_step'] = d_updates_per_train_step
    config['r_updates_per_train_step'] = r_updates_per_train_step

    # Optim
    config['r_lr'] = r_lr
    config['d_lr'] = d_lr
    config['delta'] = delta

    # Using image history
    config['use_data_history'] = use_data_history

    # Logging
    config['steps_per_log'] = steps_per_log # not currently evolved on

    # Losses (currently hard coded)
    config['self_regularization_loss'] = torch.nn.L1Loss(reduction='sum')
    config['local_adversarial_loss'] = torch.nn.CrossEntropyLoss(reduction='mean')

    return config
    

operator_dict[simgan_train_config] = {
    "inputs": [dict],
    "output": dict,
    "args": [argument_types.ArgumentType_TrainingStepsMedium,
             argument_types.ArgumentType_TrainingStepsShort,
             argument_types.ArgumentType_TrainingStepsShort,
             argument_types.ArgumentType_Int1to5,
             argument_types.ArgumentType_Int1to5,
             argument_types.ArgumentType_LearningRate,
             argument_types.ArgumentType_LearningRate,
             argument_types.ArgumentType_LearningRate,
             argument_types.ArgumentType_Bool]
    }