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
import codes.utilities.simgan_loss as simgan_loss

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
                        d_local_lr,
                        optimizer,
                        self_regularization_loss_weight,
                        minimax_loss_weight,
                        wasserstein_loss_weight,
                        local_minimax_loss_weight,
                        local_wasserstein_loss_weight,
                        local_section_size,
                        data_history_weight,
                        dragan_weight,
                        wgan_gp_weight,
                        initializer,
                        steps_per_log=100,
                        save_every=1000):
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

    # Optimizer
    config['r_lr'] = r_lr
    config['d_lr'] = d_lr 
    config['d_local_lr'] = d_local_lr

    optimizer_options = ['adam', 'rmsprop']
    ith_option = optimizer%len(optimizer_options)
    config['optimizer'] = optimizer_options[ith_option]

    # Logging
    config['steps_per_log'] = steps_per_log # not currently evolved on

    # Symbolic Regression for Losses
    config['self_regularization_loss'] = torch.nn.L1Loss(reduction='sum')
    config['self_regularization_loss_weight'] = self_regularization_loss_weight
    config['minimax_loss'] = simgan_loss.get_loss_function('minimax')
    config['minimax_loss_weight'] = minimax_loss_weight
    config['wasserstein_loss'] = simgan_loss.get_loss_function('wasserstein')
    config['wasserstein_loss_weight'] = wasserstein_loss_weight

    # Always use 'local discriminator loss' to training 
    config['local_section_size'] = local_section_size
    config['local_minimax_loss_weight'] = local_minimax_loss_weight
    config['local_wasserstein_loss_weight'] = local_wasserstein_loss_weight

    # Always use image history
    config['data_history_weight'] = data_history_weight

    # Always use gradient penalties
    config['dragan'] = simgan_loss.get_gradient_penalty('dragan')
    config['dragan_weight'] = dragan_weight
    config['wgan_gp'] = simgan_loss.get_gradient_penalty('wgan-gp')
    config['wgan_gp_weight'] = wgan_gp_weight

    # Weight initialization 
    initialization_options = ['xavier', 'uniform', 'normal', 'kaiming', 'none']
    ith_option = initializer%len(initialization_options)
    config['model_init'] = initialization_options[ith_option] 

    # Save Checkpoints
    config['save_every'] = save_every

    return config
    

operator_dict[simgan_train_config] = {
    "inputs": [dict],
    "output": dict,
    "args": [argument_types.ArgumentType_TrainingSteps,
             argument_types.ArgumentType_PretrainingSteps,
             argument_types.ArgumentType_PretrainingSteps,
             argument_types.ArgumentType_Int1to5,
             argument_types.ArgumentType_Int1to5,
             argument_types.ArgumentType_LearningRate,
             argument_types.ArgumentType_LearningRate,
             argument_types.ArgumentType_LearningRate,
             argument_types.ArgumentType_Int0to100,
             argument_types.ArgumentType_Float0to1,
             argument_types.ArgumentType_Float0to1,
             argument_types.ArgumentType_Float0to1,
             argument_types.ArgumentType_Float0to1,
             argument_types.ArgumentType_Float0to1,
             argument_types.ArgumentType_Int0to25,
             argument_types.ArgumentType_Float0to1,
             argument_types.ArgumentType_Int0to25,
             argument_types.ArgumentType_Int0to25,
             argument_types.ArgumentType_Int0to100]
    }