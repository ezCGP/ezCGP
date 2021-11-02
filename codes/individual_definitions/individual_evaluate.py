'''
root/individual_definitions/individualstandard_evaluate.py

Overview:
Super basic. All we do is define a class that has a single method: evaluate().
The method should take in IndividualMaterial (the thing it needs to evaluate), IndividualDefinition (guide for how to evaluate), and the data.

A coding law we use is that blocks will take in and output these 3 things:
    * training_datalist
    * validating_datalist
    * supplements
Sometimes those things can be None, but they should still always be used.
Training + validating datalist are mostly used for when we have multiple blocks and we want to pass
the same data types from one block to the next.
The exception comes at the last block; we mostly aways assume that we no longer car about the datalist,
and only want what is in supplements.
'''

### packages
from copy import deepcopy
from abc import ABC, abstractmethod

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from data.data_tools.ezData import ezData
from data.data_tools.simganData import SimGANDataset
from codes.genetic_material import IndividualMaterial
#from codes.individual_definitions.individual_definition import IndividualDefinition #circular dependecy
from codes.utilities.custom_logging import ezLogging
from codes.utilities.gan_tournament_selection import get_graph_ratings


def deepcopy_decorator(func):
    '''
    deepcopy the original datalist so that nothing inside individual_evaluate can change the data

    always deepcopy unless the data has a do_not_deepcopy attribute and it is true
    '''
    def inner(self,
              indiv_material,
              indiv_def,
              training_datalist,
              validating_datalist=None,
              supplements=None):
        new_training_datalist = []
        for data in training_datalist:
            if (hasattr(data, 'do_not_deepcopy')) and (data.do_not_deepcopy):
                new_training_datalist.append(data)
            else:
                new_training_datalist.append(deepcopy(data))

        if validating_datalist is not None:
            new_validating_datalist = []
            for data in validating_datalist:
                if (hasattr(data, 'do_not_deepcopy')) and (data.do_not_deepcopy):
                    new_validating_datalist.append(data)
                else:
                    new_validating_datalist.append(deepcopy(data))
        else:
            new_validating_datalist = None
        
        func(self,
             indiv_material,
             indiv_def,
             new_training_datalist,
             new_validating_datalist,
             supplements)

    return inner



class IndividualEvaluate_Abstract(ABC):
    def __init__(self):
        pass


    @abstractmethod
    def evaluate(self,
                 indiv_material: IndividualMaterial,
                 indiv_def, #: IndividualDefinition,
                 training_datalist: ezData,
                 validating_datalist: ezData=None,
                 supplements=None):
        pass


    def standard_evaluate(self,
                          indiv_id,
                          block_index,
                          block_def,
                          block_material,
                          training_datalist,
                          validating_datalist=None,
                          supplements=None,
                          apply_deepcopy=True):
        '''
        We've noted that many blocks can have slight variations for what they send to evaluate and how it is received back
        BUT there are still a lot of the same code used in each. So we made this method that should be universal to all
        blocks, and then each class can have their own custom evaluate() method where they use this universal standard_evaluate()

        Also always true:
            training_datalist, validating_datalist, supplements = block_material.output
        '''
        if apply_deepcopy:
            input_args = [deepcopy(training_datalist), deepcopy(validating_datalist), supplements]
        else:
            input_args = [training_datalist, validating_datalist, supplements]

        if block_material.need_evaluate:
            ezLogging.info("%s - Sending to %ith BlockDefinition %s to Evaluate" % (indiv_id, block_index, block_def.nickname))
            block_def.evaluate(block_material, *input_args)
        else:
            ezLogging.info("%s - Didn't need to evaluate %ith BlockDefinition %s" % (indiv_id, block_index, block_def.nickname))



class IndividualEvaluate_Standard(IndividualEvaluate_Abstract):
    '''
    for loop over each block; evaluate, take the output, and pass that in as the input to the next block
    check for dead blocks (errored during evaluation) and then just stop evaluating. Note, the remaining blocks
    should continue to have the need_evaluate flag as True.
    '''
    def __init__(self):
        pass


    @deepcopy_decorator
    def evaluate(self,
                 indiv_material: IndividualMaterial,
                 indiv_def, #: IndividualDefinition,
                 training_datalist: ezData,
                 validating_datalist: ezData=None,
                 supplements=None):
        for block_index, (block_material, block_def) in enumerate(zip(indiv_material.blocks, indiv_def.block_defs)):
            self.standard_evaluate(indiv_material.id,
                                   block_index,
                                   block_def,
                                   block_material,
                                   training_datalist,
                                   validating_datalist)
            training_datalist, validating_datalist, supplements = block_material.output
            if block_material.dead:
                indiv_material.dead = True
                indiv_material.output = [None]
                return

        indiv_material.output = block_material.output[-1]



class IndividualEvaluate_wAugmentorPipeline_wTensorFlow(IndividualEvaluate_Abstract):
    '''
    Here we are assuming that our datalist will have at least these 2 ezData instances:
        * ezData_Images
        * ezData_Augmentor

    It is also assumed that ezData_Images is HUGE, so we do not want to pass this huge thing
    into every block for evaluation, and so down the road it won't get saved to the individual
    in it's block_material.output
    Instead we only want to pass the ezData_Augmentor to blocks that handle 'preprocessing' or
    'data augmentation'.
    '''
    def __init__(self):
        pass


    @deepcopy_decorator
    def evaluate(self,
                 indiv_material: IndividualMaterial,
                 indiv_def, #IndividualDefinition,
                 training_datalist: ezData,
                 validating_datalist: ezData,
                 supplements=None):
        '''
        We only want to pass in the 'pipeline' of the data if the block does 'data augmentation' or 'data preprocessing'.

        First find the index in datalist for our ezData_Augmentor. Assume the indices are the same for training +
        validating datalists
        '''
        from data.data_tools.ezData import ezData_Augmentor

        augmentor_instance_index = None
        for i, data_instance in enumerate(training_datalist):
            if isinstance(data_instance, ezData_Augmentor):
                augmentor_instance_index = i
                break
        if augmentor_instance_index is None:
            ezLogging.error("No ezData_Augmentor instance found in training_datalist")
            exit()

        for block_index, (block_material, block_def) in enumerate(zip(indiv_material.blocks, indiv_def.block_defs)):
            if ('augment' in block_def.nickname.lower()) or ('preprocess' in block_def.nickname.lower()):
                temp_training_datalist = [training_datalist[augmentor_instance_index]]
                temp_validating_datalist = [validating_datalist[augmentor_instance_index]]
                self.standard_evaluate(indiv_material.id,
                                       block_index,
                                       block_def,
                                       block_material,
                                       temp_training_datalist,
                                       temp_validating_datalist)
                temp_training_datalist, temp_validating_datalist, _ = block_material.output
                training_datalist[augmentor_instance_index] = temp_training_datalist[0]
                validating_datalist[augmentor_instance_index] = temp_validating_datalist[0]

            elif ('tensorflow' in block_def.nickname.lower()) or ('tfkeras' in block_def.nickname.lower()):
                self.standard_evaluate(indiv_material.id,
                                       block_index,
                                       block_def,
                                       block_material,
                                       training_datalist,
                                       validating_datalist)
                #_, _, indiv_material.output = block_material.output

            else:
                self.standard_evaluate(indiv_material.id,
                                       block_index,
                                       block_def,
                                       block_material,
                                       training_datalist,
                                       validating_datalist)
                training_datalist, validating_datalist, _ = block_material.output

            if block_material.dead:
                indiv_material.dead = True
                indiv_material.output = [None]
                return

        indiv_material.output = block_material.output[-1]



class IndividualEvaluate_wAugmentorPipeline_wTensorFlow_OpenCloseGraph(IndividualEvaluate_Abstract):
    '''
    Similar to IndividualEvaluate_wAugmentorPipeline_wTensorFlow but we are going to pass supplemental info
    between TransferLearning Block and TensorFlow Block ie NN graph, first and last layer of downloaded model, etc
    '''
    def __init__(self):
        pass


    @deepcopy_decorator
    def evaluate(self,
                 indiv_material: IndividualMaterial,
                 indiv_def, #IndividualDefinition,
                 training_datalist: ezData,
                 validating_datalist: ezData,
                 supplements=None):
        '''
        placeholding
        '''
        cannot_pickle_tfkeras = True
        from data.data_tools.ezData import ezData_Augmentor

        augmentor_instance_index = None
        for i, data_instance in enumerate(training_datalist):
            if isinstance(data_instance, ezData_Augmentor):
                augmentor_instance_index = i
                break
        if augmentor_instance_index is None:
            ezLogging.error("No ezData_Augmentor instance found in training_datalist")
            exit()

        for block_index, (block_material, block_def) in enumerate(zip(indiv_material.blocks, indiv_def.block_defs)):
            if ('augment' in block_def.nickname.lower()) or ('preprocess' in block_def.nickname.lower()):
                temp_training_datalist = [training_datalist[augmentor_instance_index]]
                temp_validating_datalist = [validating_datalist[augmentor_instance_index]]
                self.standard_evaluate(indiv_material.id,
                                       block_index,
                                       block_def,
                                       block_material,
                                       temp_training_datalist,
                                       temp_validating_datalist)
                temp_training_datalist, temp_validating_datalist, _ = block_material.output
                training_datalist[augmentor_instance_index] = temp_training_datalist[0]
                validating_datalist[augmentor_instance_index] = temp_validating_datalist[0]

            elif ('transferlearning' in block_def.nickname.lower()) or \
                 ('transfer_learning' in block_def.nickname.lower()) or \
                 ('convlayers' in block_def.nickname.lower()):
                if (cannot_pickle_tfkeras) & (indiv_material[block_index+1].need_evaluate):
                    # then we had to delete the tf.keras.model in supplements index of block_material.output, so we have to re-eval
                    block_material.need_evaluate = True
                temp_training_datalist = [training_datalist[augmentor_instance_index]]
                temp_validating_datalist = [validating_datalist[augmentor_instance_index]]
                self.standard_evaluate(indiv_material.id,
                                       block_index,
                                       block_def,
                                       block_material,
                                       temp_training_datalist,
                                       temp_validating_datalist)
                # place existing tf.keras.model from transfer learning step into supplements
                temp_training_datalist, temp_validating_datalist, supplements = block_material.output
                training_datalist[augmentor_instance_index] = temp_training_datalist[0]
                validating_datalist[augmentor_instance_index] = temp_validating_datalist[0]
                if cannot_pickle_tfkeras:
                    # output is a tuple so can't directly change an element inplace
                    training_output, validating_output, supplements = block_material.output
                    block_material.output = (training_output, validating_output, None)

            elif ('tensorflow' in block_def.nickname.lower()) or \
                 ('tfkeras' in block_def.nickname.lower()) or \
                 ('denselayers' in block_def.nickname.lower()):
                self.standard_evaluate(indiv_material.id,
                                       block_index,
                                       block_def,
                                       block_material,
                                       training_datalist,
                                       validating_datalist,
                                       supplements,
                                       apply_deepcopy=False)
                #_, _, indiv_material.output = block_material.output

            else:
                self.standard_evaluate(indiv_material.id,
                                       block_index,
                                       block_def,
                                       block_material,
                                       training_datalist,
                                       validating_datalist)
                training_datalist, validating_datalist, _ = block_material.output

            if block_material.dead:
                indiv_material.dead = True
                indiv_material.output = [None]
                return
        indiv_material.output = block_material.output[-1]


class IndividualEvaluate_SimGAN(IndividualEvaluate_Abstract):
    '''
    for loop over each block; evaluate, take the output, and pass that in as the input to the next block
    check for dead blocks (errored during evaluation) and then just stop evaluating. Note, the remaining blocks
    should continue to have the need_evaluate flag as True.
    '''
    def __init__(self):
        pass

    def evaluate_block(self,
                       indiv_material,
                       block_index,
                       block_def,
                       block_material,
                       training_datalist,
                       validating_datalist,
                       supplements=None):
        '''
        Generalized evaluate method since 
        '''
        if block_material.need_evaluate:
            ezLogging.info("%s - Sending to %ith BlockDefinition %s to Evaluate" % (indiv_material.id, block_index, block_def.nickname))
            block_def.evaluate(block_material, training_datalist, validating_datalist, supplements)
            if block_material.dead:
                indiv_material.dead = True
                return
            else:
                pass
        else:
            ezLogging.info("%s - Didn't need to evaluate %ith BlockDefinition %s" % (indiv_material.id, block_index, block_def.nickname))


    def evaluate(self,
                 indiv_material,
                 indiv_def, 
                 training_datalist,
                 validating_datalist,
                 supplements=None):
        '''
        Because the Refiner and Discriminator are two seperate blocks but require one another for their loss functions, they must be run together
        So we will take the refiner graph and input it into the discriminator block and train/evaluate it there.
        '''
        # This essentially builds the refiner and discriminator networks and sets everything up
        supplements = []
        for block_index, (block_material, block_def) in enumerate(zip(indiv_material.blocks, indiv_def.block_defs)):
            if (block_def.nickname == 'refiner_block') and (indiv_material[block_index+1].need_evaluate or indiv_material[block_index+2].need_evaluate):
                # If we are in the refiner block and the discriminator block or train config needs to be reevaluated, then we also need to reevalute the refiner
                block_material.need_evaluate = True
                self.evaluate_block(indiv_material,
                                    block_index,
                                    block_def,
                                    block_material,
                                    training_datalist,
                                    validating_datalist,
                                    supplements
                                    )
                training_datalist, validating_datalist, supplements = block_material.output
            elif (block_def.nickname == 'discriminator_block') and (indiv_material[block_index+1].need_evaluate):
                # If we are in the discriminator block and the train config needs to be reevaluated, then we also need to reevalute the discriminator
                block_material.need_evaluate = True
                self.evaluate_block(indiv_material,
                                    block_index,
                                    block_def,
                                    block_material,
                                    training_datalist,
                                    validating_datalist,
                                    supplements
                                    )
                training_datalist, validating_datalist, supplements = block_material.output
            else:
                self.evaluate_block(indiv_material,
                                    block_index,
                                    block_def,
                                    block_material,
                                    training_datalist,
                                    validating_datalist,
                                    supplements)
                training_datalist, validating_datalist, supplements = block_material.output
        
        # Train the refiner and discriminator
        untrained_refiner, untrained_discriminator, train_config = supplements
        refiners, discriminators = self.train_graph(indiv_material, training_datalist[0], validating_datalist[0], 
            untrained_refiner, untrained_discriminator, train_config)

        # Now do tournament selection to pick the best refiner/discriminator networks from the training process
        # TODO: consider finding a way to replace this with a convergence metric or something
        ezLogging.debug("Finding best refiner and discriminator for individual")
        refiner_ratings, discriminator_ratings = get_graph_ratings(refiners, discriminators, validating_datalist[0], train_config['device'])
        best_refiner = refiners[refiner_ratings['r'].argmax()]
        best_discriminator = discriminators[discriminator_ratings['r'].argmax()]
        
        indiv_material.output = (best_refiner, best_discriminator)


    def pretrain_networks(self, train_data, R, D, train_config, opt_R, opt_D):
        '''
        Pretrain the refiner to learn the identity function and discriminator to learn the difference
        between simulated and real data.
        '''
        import torch
        from torch import Tensor, LongTensor

        # Pretrain Refiner to learn the identity function
        ezLogging.debug("Pretraining refiner for %i steps" % (train_config['r_pretrain_steps']))
        ezLogging.debug("Pretraining discriminator for %i steps"  % (train_config['d_pretrain_steps']))
        for i in range(train_config['r_pretrain_steps']):
            opt_R.zero_grad()
            
            # Load data
            simulated, _ = train_data.simulated_loader.__iter__().next()
            simulated = Tensor(simulated).to(train_config['device'])

            # Run refiner and get self_regularization loss
            refined = R(simulated)
            r_loss = train_config['self_regularization_loss'](simulated, refined)
            r_loss = torch.mul(r_loss, train_config['delta'])

            # Compute the gradients and backprop
            r_loss.backward()
            opt_R.step()

            # log every `steps_per_log` steps
            if ((i+1) % train_config['steps_per_log'] == 0) or (i == train_config['r_pretrain_steps'] - 1):
                print('[%d/%d] (R)reg_loss: %.4f' % (i+1, train_config['r_pretrain_steps'], r_loss.data.item()))
        
        # Pretrain Discriminator (basically to learn the difference between simulated and real data)
        try:
            for i in range(train_config['d_pretrain_steps']):
                opt_D.zero_grad()

                # Get data
                real, labels_real = train_data.real_loader.__iter__().next()
                real = Tensor(real).to(train_config['device'])
                labels_real = LongTensor(labels_real).to(train_config['device'])

                simulated, labels_refined = train_data.simulated_loader.__iter__().next()
                simulated = Tensor(simulated).to(train_config['device'])
                labels_refined = LongTensor(labels_refined).to(train_config['device'])

                # Run the real batch through discriminator and calc loss
                pred_real = D(real)
                d_loss_real = train_config['local_adversarial_loss'](pred_real, labels_real)

                # Run the refined batch through discriminator and calc loss
                refined = R(simulated)
                pred_refined = D(refined)
                d_loss_ref = train_config['local_adversarial_loss'](pred_refined, labels_refined)

                # Compute the gradients.
                d_loss = d_loss_real + d_loss_ref
                d_loss.backward()

                # Backpropogate the gradient through the discriminator.
                opt_D.step()

                # log every `steps_per_log` steps
                if ((i+1) % train_config['steps_per_log'] == 0) or (i == train_config['d_pretrain_steps'] - 1):
                    print('[%d/%d] (D)real_loss: %.4f, ref_loss: %.4f' % (i+1, train_config['d_pretrain_steps'], 
                        d_loss_real.data.item(), d_loss_ref.data.item()))
        except Exception as e:
            import pdb; pdb.set_trace()
            print('Failed to run discriminator')

    
    # TODO: see if we should be utilizing validation data
    # TODO: find a better way of picking networks to save than just every n steps
    # TODO: change save_every to 200 or 100
    def train_graph(self, indiv_material, train_data, validation_data, R, D, train_config, save_every=50):
        '''
        Train the refiner and discriminator of the SimGAN, return a refiner and discriminator pair for every 'save_every' training steps
        '''
        import torch

        ezLogging.debug("%s - Training Graph - %i batch size, %i steps" % (indiv_material.id,
                                                                           train_data.batch_size,
                                                                           train_config['train_steps']))
        opt_R = torch.optim.Adam(R.parameters(), lr=train_config['r_lr'], betas=(0.5,0.999))
        opt_D = torch.optim.Adam(D.parameters(), lr=train_config['d_lr'], betas=(0.5,0.999))
        self.pretrain_networks(train_data, R, D, train_config, opt_R, opt_D)

        # TRAINING #
        r_losses = []
        d_losses = []
        refiners = []
        discriminators = []
        for step in range(train_config['train_steps']):
            # ========= Train the Refiner ========= 
            total_r_loss = 0.0
            total_r_loss_reg = 0.0
            total_r_loss_adv = 0.0
            for index in range(train_config['r_updates_per_train_step']):
                opt_R.zero_grad()

                # Load data
                simulated, _ = train_data.simulated_loader.__iter__().next()
                simulated = torch.Tensor(simulated).to(train_config['device'])
                real_labels = torch.zeros(simulated.shape[0], dtype=torch.long).to(train_config['device'])

                # Run refiner and get self_regularization loss
                refined = R(simulated)
                r_loss_reg = train_config['delta'] * train_config['self_regularization_loss'](simulated, refined) 
                # Run discriminator on refined data and get adversarial loss
                d_pred = D(refined)
                r_loss_adv = train_config['local_adversarial_loss'](d_pred, real_labels) # want discriminator to think they are real

                # Compute the gradients and backprop
                r_loss = r_loss_reg + r_loss_adv
                r_loss.backward()
                opt_R.step()

                # Update loss records
                total_r_loss += r_loss
                total_r_loss_reg += r_loss_reg
                total_r_loss_adv += r_loss_adv
                
            # track avg. refiner losses
            mean_r_loss = total_r_loss / train_config['r_updates_per_train_step']
            r_losses.append(mean_r_loss)

            # ========= Train the Discriminator ========= 
            total_d_loss = 0.0
            total_d_loss_real = 0.0
            total_d_loss_ref = 0.0
            for index in range(train_config['d_updates_per_train_step']):
                opt_D.zero_grad()

                # Get data
                real, labels_real = train_data.real_loader.__iter__().next()
                real = torch.Tensor(real).to(train_config['device'])
                labels_real = torch.LongTensor(labels_real).to(train_config['device'])

                simulated, labels_refined = train_data.simulated_loader.__iter__().next()
                simulated = torch.Tensor(simulated).to(train_config['device'])
                labels_refined = torch.LongTensor(labels_refined).to(train_config['device'])

                # import pdb; pdb.set_trace()
                # Run the real batch through discriminator and calc loss
                pred_real = D(real)
                d_loss_real = train_config['local_adversarial_loss'](pred_real, labels_real)

                # Run the refined batch through discriminator and calc loss
                refined = R(simulated)

                # use a history of refined signals
                # TODO: investigate keeping track of loss of old refined from history buffer and new refined seperately
                d_loss_ref = None
                if train_config['use_data_history']:
                    if not train_data.data_history_buffer.is_empty():
                        refined_hist = train_data.data_history_buffer.get()
                        refined_hist = torch.Tensor(refined_hist).to(train_config['device'])
                        pred_refined_size = len(refined) - len(refined_hist)

                        pred_refined = D(refined[:pred_refined_size])
                        pred_refined_hist = D(refined_hist)

                        d_loss_ref = train_config['local_adversarial_loss'](pred_refined, labels_refined[:pred_refined_size])
                        d_loss_ref_hist = train_config['local_adversarial_loss'](pred_refined_hist, labels_refined[pred_refined_size:])
                        d_loss_ref += d_loss_ref_hist
                    else:
                        pred_refined = D(refined)
                        d_loss_ref = train_config['local_adversarial_loss'](pred_refined, labels_refined)

                    train_data.data_history_buffer.add(refined.cpu().data.numpy())
                else:
                    pred_refined = D(refined)
                    d_loss_ref = train_config['local_adversarial_loss'](pred_refined, labels_refined)

                # Compute the gradients.
                d_loss = d_loss_real + d_loss_ref
                d_loss.backward()

                # Backpropogate the gradient through the discriminator.
                opt_D.step()

                total_d_loss += d_loss
                total_d_loss_real += d_loss_real
                total_d_loss_ref = d_loss_ref

            # track avg. discriminator losses
            mean_d_loss = total_d_loss / train_config['d_updates_per_train_step']
            d_losses.append(mean_d_loss)
                
            # log every `steps_per_log` steps
            if ((step+1) % train_config['steps_per_log'] == 0) or (step == train_config['train_steps'] - 1):
                print('[%d/%d] ' % (step + 1, train_config['train_steps']))

                mean_r_loss_reg = total_r_loss_reg / train_config['r_updates_per_train_step']
                mean_r_loss_adv = total_r_loss_adv / train_config['r_updates_per_train_step']
                print('(R) mean_refiner_total_loss: %.4f mean_r_loss_reg: %.4f, mean_r_loss_adv: %.4f'
                    % (mean_r_loss.data.item(), mean_r_loss_reg.data.item(), mean_r_loss_adv.data.item()))

                mean_d_loss_real = total_d_loss_real / train_config['d_updates_per_train_step']
                mean_d_loss_ref = total_d_loss_ref / train_config['d_updates_per_train_step']
                print('(D) mean_discriminator_loss: %.4f mean_d_real_loss: %.4f, mean_d_ref_loss: %.4f' 
                    % (mean_d_loss.data.item(), mean_d_loss_real.data.item(), mean_d_loss_ref.data.item()))
        
            # Save every `save_every` steps:
            if ((step+1) % save_every == 0):
                refiners.append(deepcopy(R))
                discriminators.append(deepcopy(D))
        
        return refiners, discriminators
