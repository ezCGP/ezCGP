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
import importlib

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

        output = func(self,
                      indiv_material,
                      indiv_def,
                      new_training_datalist,
                      new_validating_datalist,
                      supplements)
        return output

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
    def __init__(self,
                 gradient_penalty='dragan',
                 loss=None,
                 penalty_constant=10):
        # import torch to be used throughout IndividualEvaluate class
        globals()['torch'] = importlib.import_module('torch')

        from codes.utilities.simgan_loss import get_gradient_penalty, get_loss_function, xavier_init
        self.gradient_penalty = get_gradient_penalty(gradient_penalty)
        if gradient_penalty == "dragan":
            self.model_init = xavier_init
        else:
            self.model_init = None
        self.loss_function = get_loss_function(loss) # TODO wasserstein
        self.penalty_constant = penalty_constant


    @deepcopy_decorator
    def evaluate(self,
                 indiv_material: IndividualMaterial,
                 indiv_def, #IndividualDefinition,
                 training_datalist: ezData,
                 validating_datalist: ezData,
                 supplements=None):
        '''
        Because the Refiner and Discriminator are two seperate blocks but require one another for their loss functions,
        they must be run together, so each will be evaluated to build the graphs but then trained here.
        '''
        from codes.utilities.gan_tournament_selection import get_graph_ratings

        # Build the Graphs/Networks
        block_outputs = []
        # going to go in reverse order since we need an flag fromTrain_Config to guide evaluation of discriminator
        for block_index in range(indiv_def.block_count-1, -1, -1):
            block_material = indiv_material.blocks[block_index]
            block_def = indiv_def.block_defs[block_index]
            if block_material.need_evaluate:
                ezLogging.info("%s - Sending to %ith BlockDefinition %s to Evaluate" % (indiv_material.id, block_index, block_def.nickname))
                block_def.evaluate(block_material, training_datalist, None, None)
                if block_material.dead:
                    indiv_material.dead = True
                    return None
                else:
                    pass
            else:
                ezLogging.info("%s - Didn't need to evaluate %ith BlockDefinition %s" % (indiv_material.id, block_index, block_def.nickname))

            # just a way so we can run through evaluation of each individual quickly
            TESTING_HACK = False
            if (TESTING_HACK) and ('train_config' in block_def.nickname):
                block_material.output[0]['train_steps'] = 10
                block_material.output[0]['r_pretrain_steps'] = 10
                block_material.output[0]['d_pretrain_steps'] = 10
                block_material.output[0]['d_updates_per_train_step'] = 10
                block_material.output[0]['r_updates_per_train_step'] = 10
                block_material.output[0]['steps_per_log'] = 10
                block_material.output[0]['save_every'] = 10

            # adding deepcopy to make sure we can save the 'untrained' states in block.output and that they don't get overwritten in training
            block_outputs += deepcopy(block_material.output)

            if block_index == 2:
                assert("train" in block_def.nickname and "config" in block_def.nickname), "Our assumption that index 2 is train_config block is wrong!"
                if (not hasattr(indiv_material[1], 'train_local_loss')) or (indiv_material[1].train_local_loss != block_material.output[0]['train_local_loss']):
                    indiv_material[1].train_local_loss = block_material.output[0]['train_local_loss']
                    indiv_material[1].need_evaluate = True
                if (not hasattr(indiv_material[1], 'local_section_size')) or (indiv_material[1].local_section_size != block_material.output[0]['local_section_size']):
                    indiv_material[1].local_section_size = block_material.output[0]['local_section_size']
                    if indiv_material[1].train_local_loss:
                        # even if 'local_section_size' changes, it won't matter if we are not training local loss
                        indiv_material[1].need_evaluate = True

        # Adding GPU meta data info
        #https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
        if torch.cuda.is_available():
            for gpu_device in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(gpu_device)
                if gpu_name is not None:
                    ezLogging.debug("%s - Using GPU #%i: %s" % (indiv_material.id, gpu_device, gpu_name))

        train_config, untrained_discriminator, untrained_local_discriminator, untrained_refiner = block_outputs
        untrained_refiner.to(train_config['device'])
        untrained_discriminator.to(train_config['device'])
        if untrained_local_discriminator:
            untrained_local_discriminator.to(train_config['device'])

        # if using dragan gradient penalty, init with xavier per their implementation
        # https://github.com/kodalinaveen3/DRAGAN
        if hasattr(self, 'model_init') and self.model_init is not None:
            self.model_init(untrained_refiner)
            self.model_init(untrained_discriminator)
            if untrained_local_discriminator:
                self.model_init(untrained_local_discriminator)

        if train_config['optimizer'] == 'adam':
            opt_R = torch.optim.Adam(untrained_refiner.parameters(), lr=train_config['r_lr'], betas=(0.5,0.999))
            opt_D = torch.optim.Adam(untrained_discriminator.parameters(), lr=train_config['d_lr'], betas=(0.5,0.999))
            opt_D_local = None
            if untrained_local_discriminator:
                opt_D_local = torch.optim.Adam(untrained_local_discriminator.parameters(), lr=train_config['d_lr'], betas=(0.5,0.999))
        elif train_config['optimizer'] == 'rmsprop':
            opt_R = torch.optim.RMSprop(untrained_refiner.parameters(), lr=train_config['r_lr'])
            opt_D = torch.optim.RMSprop(untrained_discriminator.parameters(), lr=train_config['d_lr'])
            opt_D_local = None
            if untrained_local_discriminator:
                opt_D_local = torch.optim.RMSprop(untrained_local_discriminator.parameters(), lr=train_config['d_lr'])
        else:
            ezLogging.critical("%s - Reached an invalid value for Network Optimizer: %s" % (indiv_material.id, train_config['optimizer']))

        # NOTE that block_def.evaluate() will have their own try/except to catch errors and kill the individual and return None
        try:
            self.pretrain_networks(training_datalist[0],
                                   untrained_refiner,
                                   untrained_discriminator,
                                   untrained_local_discriminator,
                                   train_config,
                                   opt_R,
                                   opt_D,
                                   opt_D_local)
        except Exception as err:
            ezLogging.critical("%s - PreTrain Graph; Failed: %s" % (indiv_material.id, err))
            indiv_material.dead = True
            #import pdb; pdb.set_trace()
            return None

        try:
            # Train the refiner and discriminator
            refiners, discriminators = self.train_graph(indiv_material,
                                                        training_datalist[0],
                                                        validating_datalist[0],
                                                        untrained_refiner,
                                                        untrained_discriminator,
                                                        untrained_local_discriminator,
                                                        train_config,
                                                        opt_R,
                                                        opt_D,
                                                        opt_D_local)
        except Exception as err:
            ezLogging.critical("%s - Train Graph; Failed: %s" % (indiv_material.id, err))
            indiv_material.dead = True
            #import pdb; pdb.set_trace()
            return None

        # Now do tournament selection to pick the best refiner/discriminator networks from the training process
        # TODO: consider finding a way to replace this with a convergence metric or something
        ezLogging.debug("Finding best refiner and discriminator for individual")
        refiner_ratings, discriminator_ratings = get_graph_ratings(refiners,
                                                                   discriminators,
                                                                   validating_datalist[0],
                                                                   train_config['device'])
        # use idxmax instead of argmax since it is more correct that we want row label instead of index in dataframe
        best_refiner = refiners[refiner_ratings['r'].idxmax()]
        best_discriminator = discriminators[discriminator_ratings['r'].idxmax() - len(refiners)]

        # clear gpu memory
        torch.cuda.empty_cache()

        indiv_material.output = (best_refiner, best_discriminator)


    def pretrain_networks(self, train_data, R, D, D_local, train_config, opt_R, opt_D, opt_D_local):
        '''
        Pretrain the refiner to learn the identity function and discriminator to learn the difference
        between simulated and real data.
        '''

        # Pretrain Refiner to learn the identity function
        ezLogging.info("Pretraining refiner for %i steps" % (train_config['r_pretrain_steps']))
        for i in range(train_config['r_pretrain_steps']):
            opt_R.zero_grad()

            # Load data
            simulated, _ = train_data.simulated_loader.__iter__().next()
            simulated = torch.Tensor(simulated).to(train_config['device'])

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
        ezLogging.info("Pretraining discriminator for %i steps"  % (train_config['d_pretrain_steps']))
        for i in range(train_config['d_pretrain_steps']):
            opt_D.zero_grad()

            # Get data
            real, labels_real = train_data.real_loader.__iter__().next()
            real = torch.Tensor(real).to(train_config['device'])
            labels_real = torch.FloatTensor(labels_real).to(train_config['device'])

            simulated, labels_refined = train_data.simulated_loader.__iter__().next()
            simulated = torch.Tensor(simulated).to(train_config['device'])
            labels_refined = torch.FloatTensor(labels_refined).to(train_config['device'])

            # Run the real batch through discriminator and calc loss
            pred_real = D(real)
            d_loss_real = train_config['local_adversarial_loss'](pred_real.to(torch.float32), labels_real.to(torch.float32))

            # Run the refined batch through discriminator and calc loss
            refined = R(simulated)
            pred_refined = D(refined)
            d_loss_ref = train_config['local_adversarial_loss'](pred_refined.to(torch.float32), labels_refined.to(torch.float32))

            if D_local:
                real_batch_split = torch.split(real, train_config['local_section_size'], dim=2)
                # Calculate the predictions of the local section
                real_section_preds = []
                for section in real_batch_split:
                    # TODO Getting an argmax on empty sequence error on the following line
                    # We don't use argmax at all...
                    # Double check the given network and make sure its right
                    pred_real_section = D_local(section)
                    real_section_preds.append(pred_real_section)

                # Stack and average the predictions together to get the "overall" prediction of the sample
                preds_real_agg = torch.stack(real_section_preds)
                pred_real_local = torch.mean(preds_real_agg, dim=0)
                d_loss_real_local = train_config['local_adversarial_loss'](pred_real_local.to(torch.float32), labels_real.to(torch.float32))
                d_loss_real += d_loss_real_local

                # Continue the same process on the refined samples
                ref_batch_split = torch.split(refined, train_config['local_section_size'], dim=2)
                ref_section_preds = []
                for section in ref_batch_split:
                    pred_ref_section = D_local(section)
                    ref_section_preds.append(pred_ref_section)

                preds_ref_agg = torch.stack(ref_section_preds)
                pred_ref_local = torch.mean(preds_ref_agg, dim=0)
                d_loss_ref_local = train_config['local_adversarial_loss'](pred_ref_local.to(torch.float32), labels_refined.to(torch.float32))
                d_loss_ref += d_loss_ref_local


            # Compute the gradients.
            d_loss = d_loss_real + d_loss_ref

            # Gradient Penalty
            if hasattr(self, 'gradient_penalty') and self.gradient_penalty is not None:
                d_loss += self.gradient_penalty(D,
                                                real,
                                                refined,
                                                train_data.batch_size,
                                                self.penalty_constant,
                                                cuda=True,
                                                device=train_config['device'])

            # Backpropogate the gradient through the discriminator.
            d_loss.backward()
            opt_D.step()

            if opt_D_local:
                opt_D_local.step()

            # log every `steps_per_log` steps
            if ((i+1) % train_config['steps_per_log'] == 0) or (i == train_config['d_pretrain_steps'] - 1):
                print('[%d/%d] (D)real_loss: %.4f, ref_loss: %.4f' % (i+1, train_config['d_pretrain_steps'],
                    d_loss_real.data.item(), d_loss_ref.data.item()))


    # TODO: see if we should be utilizing validation data
    # TODO: find a better way of picking networks to save than just every n steps
    # TODO: change save_every to 200 or 100
    def train_graph(self, indiv_material, train_data, validation_data, R, D, D_local, train_config, opt_R, opt_D, opt_D_local):
        '''
        Train the refiner and discriminator of the SimGAN, return a refiner and discriminator pair for every train_config['save_every'] training steps
        '''

        ezLogging.info("%s - Training Graph - %i batch size, %i steps" % (indiv_material.id,
                                                                           train_data.batch_size,
                                                                           train_config['train_steps']))

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
                real_labels = torch.zeros(simulated.shape[0], dtype=torch.float).to(train_config['device'])

                # Run refiner and get self_regularization loss
                refined = R(simulated)
                r_loss_reg = train_config['delta'] * train_config['self_regularization_loss'](simulated, refined)
                # Run discriminator on refined data and get adversarial loss
                d_pred = D(refined)
                r_loss_adv = train_config['local_adversarial_loss'](d_pred.to(torch.float32), real_labels.to(torch.float32)) # want discriminator to think they are real

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
                labels_real = torch.FloatTensor(labels_real).to(train_config['device'])

                simulated, labels_refined = train_data.simulated_loader.__iter__().next()
                simulated = torch.Tensor(simulated).to(train_config['device'])
                labels_refined = torch.FloatTensor(labels_refined).to(train_config['device'])

                # import pdb; pdb.set_trace()
                # Run the real batch through discriminator and calc loss
                pred_real = D(real)
                d_loss_real = train_config['local_adversarial_loss'](pred_real.to(torch.float32), labels_real.to(torch.float32))

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

                        d_loss_ref = train_config['local_adversarial_loss'](pred_refined.to(torch.float32), labels_refined[:pred_refined_size].to(torch.float32))
                        d_loss_ref_hist = train_config['local_adversarial_loss'](pred_refined_hist.to(torch.float32), labels_refined[pred_refined_size:].to(torch.float32))
                        d_loss_ref += d_loss_ref_hist

                        if D_local:
                            ref_batch_split = torch.split(refined, train_config['local_section_size'], dim=2)
                            ref_section_preds = []
                            ref_hist_split = torch.split(refined_hist, train_config['local_section_size'], dim=2)

                            for section in ref_batch_split:
                                pred_ref_section = D_local(section[:pred_refined_size])
                                ref_section_preds.append(pred_ref_section)
                            preds_ref_agg = torch.stack(ref_section_preds)
                            pred_ref_local_curr = torch.mean(preds_ref_agg, dim=0)

                            ref_hist_preds = []
                            for section in ref_hist_split:
                                pred_ref_hist_local = D_local(section)
                                ref_hist_preds.append(pred_ref_hist_local)
                            preds_ref_hist_agg = torch.stack(ref_hist_preds)
                            pred_ref_local_hist = torch.mean(preds_ref_hist_agg, dim=0)

                            # Run the refined batch through discriminator and calc loss/accuracy
                            d_loss_ref_local = train_config['local_adversarial_loss'](pred_ref_local_curr, labels_refined[:pred_refined_size])
                            d_loss_ref_hist_local = train_config['local_adversarial_loss'](pred_ref_local_hist, labels_refined[:pred_refined_size])
                            d_loss_ref += d_loss_ref_local + d_loss_ref_hist_local
                    else:
                        pred_refined = D(refined)
                        d_loss_ref = train_config['local_adversarial_loss'](pred_refined.to(torch.float32), labels_refined.to(torch.float32))

                        if D_local:
                            real_batch_split = torch.split(real, train_config['local_section_size'], dim=2)
                            # Calculate the predictions of the local section
                            real_section_preds = []
                            for section in real_batch_split:
                                pred_real_section = D_local(section)
                                real_section_preds.append(pred_real_section)

                            # Stack and average the predictions together to get the "overall" prediction of the sample
                            preds_real_agg = torch.stack(real_section_preds)
                            pred_real_local = torch.mean(preds_real_agg, dim=0)
                            d_loss_real_local = train_config['local_adversarial_loss'](pred_real_local.to(torch.float32), labels_real.to(torch.float32))
                            d_loss_real += d_loss_real_local

                            # Continue the same process on the refined samples
                            ref_batch_split = torch.split(refined, train_config['local_section_size'], dim=2)
                            ref_section_preds = []
                            for section in ref_batch_split:
                                pred_ref_section = D_local(section)
                                ref_section_preds.append(pred_ref_section)

                            preds_ref_agg = torch.stack(ref_section_preds)
                            pred_ref_local = torch.mean(preds_ref_agg, dim=0)
                            d_loss_ref_local = train_config['local_adversarial_loss'](pred_ref_local.to(torch.float32), labels_refined.to(torch.float32))
                            d_loss_ref += d_loss_ref_local

                    train_data.data_history_buffer.add(refined.cpu().data.numpy())
                else:
                    pred_refined = D(refined)
                    d_loss_ref = train_config['local_adversarial_loss'](pred_refined.to(torch.float32), labels_refined.to(torch.float32))

                    if D_local:
                        real_batch_split = torch.split(real, train_config['local_section_size'], dim=2)
                        # Calculate the predictions of the local section
                        real_section_preds = []
                        for section in real_batch_split:
                            pred_real_section = D_local(section)
                            real_section_preds.append(pred_real_section)

                        # Stack and average the predictions together to get the "overall" prediction of the sample
                        preds_real_agg = torch.stack(real_section_preds)
                        pred_real_local = torch.mean(preds_real_agg, dim=0)
                        d_loss_real_local = train_config['local_adversarial_loss'](pred_real_local.to(torch.float32), labels_real.to(torch.float32))
                        d_loss_real += d_loss_real_local

                        # Continue the same process on the refined samples
                        ref_batch_split = torch.split(refined, train_config['local_section_size'], dim=2)
                        ref_section_preds = []
                        for section in ref_batch_split:
                            pred_ref_section = D_local(section)
                            ref_section_preds.append(pred_ref_section)

                        preds_ref_agg = torch.stack(ref_section_preds)
                        pred_ref_local = torch.mean(preds_ref_agg, dim=0)
                        d_loss_ref_local = train_config['local_adversarial_loss'](pred_ref_local.to(torch.float32), labels_refined.to(torch.float32))
                        d_loss_ref += d_loss_ref_local

                # Compute the gradients.
                d_loss = d_loss_real + d_loss_ref

                # Gradient Penalty
                if hasattr(self, 'gradient_penalty') and self.gradient_penalty is not None:
                    d_loss += self.gradient_penalty(D,
                                                    real,
                                                    refined,
                                                    train_data.batch_size,
                                                    self.penalty_constant,
                                                    cuda=True,
                                                    device=train_config['device'])


                # Backpropogate the gradient through the discriminator.
                d_loss.backward()
                opt_D.step()

                if opt_D_local:
                    opt_D_local.step()

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
            if ((step+1) % train_config['save_every'] == 0):
                ezLogging.info("Training %i/%i" % (step, train_config['train_steps']))
                refiners.append(deepcopy(R))
                discriminators.append(deepcopy(D))

        return refiners, discriminators
