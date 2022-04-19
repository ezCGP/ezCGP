'''
experimental loss function for simgan
'''
import torch
from torch import nn
from torch.nn import functional as F
import torch.autograd as autograd
from torch.nn.init import xavier_normal_


def xavier_init(model):
    for param in model.parameters():
        if len(param.size()) == 2:
            xavier_normal_(param)


def get_loss_function(loss_name):
    '''
    Determines which loss function to use given the name from the xml
        Parameters:
            loss_name (string): The name of the GAN loss function that we would like to use: {minimax, wasserstein}
 
        Returns:
            loss_function (func): A callable function that calculates and return the loss given the predictions and true labels
    '''
    if loss_name == 'minimax':
        return nn.BCEWithLogitsLoss(reduction='mean')
    elif loss_name == 'wasserstein':
        return wasserstein_loss
    else:
        return None


def get_gradient_penalty(penalty_name):
    '''
    Determines which gradient penalty function to use given the name from the xml
        Parameters:
            penalty_name (string): The name of the gradient penalty that we would like to use: {dragan, wgan-gp}
 
        Returns:
            penalty_function (func): A callable function that calculates and return the penalty calculation
    '''
    if penalty_name == 'dragan':
        return calc_dragan_gradient_penalty
    elif penalty_name == 'wgan-gp':
        return calc_wgan_gradient_penalty
    else:
        return None


def calc_wgan_gradient_penalty(netD, real_data, fake_data, batch_size=4, penalty_constant=10, cuda=True, device=None):
    '''
    Calculates the WGAN-GP gradient penalty
    https://arxiv.org/abs/1704.00028
 
        Parameters:
            netD: discriminator model
            real_data: batch of real data
            fake_data: batch of data from refiner
            batch_size: batch size used to generate noise tensor
            penalty_constant: constant defined in xml
            cuda: bool for cuda enabled
            device: device to assign
 
        Returns:
            gradient_penalty (float): Value of the penalty
    '''
    _, layers, feature_count = real_data.shape
    alpha = torch.rand(batch_size, layers, feature_count, requires_grad=True)
    if cuda:
        alpha = alpha.cuda(device)
 
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
 
    if cuda:
        interpolates = interpolates.cuda(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
 
    disc_interpolates = netD(interpolates)
 
    if cuda:
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.zeros(disc_interpolates.size()).cuda(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    else:
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.zeros(disc_interpolates.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
 
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty_constant
    return gradient_penalty


def calc_dragan_gradient_penalty(netD, real_data, fake_data, batch_size=4, penalty_constant=10, cuda=True, device=None):
    '''
    Calculates the DRAGAN gradient penalty
    https://arxiv.org/abs/1705.07215
 
        Parameters:
            netD: discriminator model
            real_data: batch of real data
            fake_data: batch of data from refiner
            batch_size: batch size used to generate noise tensor
            penalty_constant: constant defined in xml
            cuda: bool for cuda enabled
            device: device to assign
 
        Returns:
            gradient_penalty (float): Value of the penalty
    '''
    _, layers, feature_count = real_data.shape
    alpha = torch.rand(batch_size, layers, feature_count, requires_grad=True)
    if cuda:
        alpha = alpha.cuda(device)
        interpolates = alpha * real_data + ((1 - alpha) * (fake_data + 0.5 * fake_data.std() * torch.rand(fake_data.size()).cuda(device)))
        interpolates = interpolates.cuda(device)
    else:
        interpolates = alpha * real_data + ((1 - alpha) * (fake_data + 0.5 * fake_data.std() * torch.rand(fake_data.size())))
    interpolates = autograd.Variable(interpolates, requires_grad=True)
 
    disc_interpolates = netD(interpolates)
 
    if cuda:
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.zeros(disc_interpolates.size()).cuda(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    else:
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.zeros(disc_interpolates.size()),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
 
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty_constant
    
    return gradient_penalty


def wasserstein_loss(pred, labels):
    '''
    Calculates the wasserstein loss for the refiner and part of the descriminator loss
    https://arxiv.org/abs/1701.07875
 
        Parameters:
            pred (ndarray): tensor from discriminator
            labels (ndarray): Unused
 
        Returns:
            loss (float): Value of the loss
    '''
    loss = -torch.mean(pred)
    return loss