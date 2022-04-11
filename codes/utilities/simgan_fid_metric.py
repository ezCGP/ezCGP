'''
root/codes/utilities/simgan_fid_metric.py

Frechet Inception Distance metric to be used for simgan performance
https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch
'''

### packages
import glob
import numpy as np
import scipy.linalg
import torch
from torchvision import models
from torchvision import __version__ as torchvision_version
import torch.nn as nn
from torchvision import transforms
from numpy.random import random
from scipy.linalg import sqrtm

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from codes.utilities.custom_logging import ezLogging


def get_model(offline_mode=False, model_name='inception_v3'):
    if offline_mode:
        '''
        See Issue #268
        Check respective model file to find .pth download link
        https://github.com/pytorch/vision/tree/main/torchvision/models
        '''
        # note: glob relative to ezCGP repo root, not this file
        if len(glob.glob("./test_area/%s*" % model_name)) > 0:
            # get model file
            if model_name == 'inception_v3':
                if (int(torchvision_version.split(".")[0]) == 0) and\
                   (int(torchvision_version.split(".")[1]) <= 9):
                    # everything including and under v0.9.0
                    model_weights_file = "./test_area/inception_v3_google-1a9a5a14.pth"
                else:
                    model_weights_file = "./test_area/inception_v3_google-0cc3c7bd.pth"
            else:
                model_weights_file = glob.glob("./test_area/%s*" % model_name)[-1]

            model_weights = torch.load(model_weights_file)
            model = models.__dict__[model_name](pretrained=False)
            model.load_state_dict(model_weights)
            del model_weights, model_weights_file
            ezLogging.info("In 'offline_mode' and successfully found .pth file for %s" % model_name)
        else:
            raise Exception("In 'offline_mode' but couldn't find .pth file for %s" % model_name)

    else:
        model = torch.hub.load('pytorch/vision:v0.8.0', model_name, pretrained=True)
        ezLogging.info("Successfully downloaded or found in cache pretrained %s" % model_name)

    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    model.avgpool.register_forward_hook(get_features('feats'))
    model.eval()

    return model, features


def process(input):
    batch_size = input.shape[0]
    z = torch.zeros((batch_size, 23))
    h = torch.hstack([input, input, input, z])
    
    output = torch.stack([h for _ in range(299)])
    output = output.permute(1,0,2)
    
    output = torch.stack([output, output, output])
    output = output.permute(1,0,2,3)
    return output


def get_activations(model, features, inputs):
    inputs = process(inputs)
        
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(inputs)
        return features['feats'].squeeze()


def calculate_fid(model, features, ref, real):
    '''
    adapted from
    https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch
    '''
    act1 = get_activations(model, features, ref).cpu().data
    act2 = get_activations(model, features, real).cpu().data

    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = ((mu1 - mu2)**2).sum()

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid.item()


def get_fid_scores(refiners, validation_data, offline_mode):
    all_real = validation_data.real_raw
    all_sim = validation_data.simulated_raw
    
    batch_size = min([all_real.shape[0], all_sim.shape[0]])
   
    model, features = get_model(offline_mode)
    chosen_sim = torch.tensor(all_sim[:batch_size, :, :], dtype=torch.float, device='cpu')
    chosen_real = torch.tensor(all_real[:batch_size, :, :], dtype=torch.float, device='cpu')

    fid_scores = []
    for i, R in enumerate(refiners):
        ref = R(chosen_sim)
        fid = calculate_fid(model, features, ref.squeeze(), chosen_real.squeeze())
        fid_scores.append(fid)

    return fid_scores


def test():
    model, features = get_model()
    a = torch.ones([128, 92])
    b = torch.randn([128, 92])
    print(calculate_fid(model, features, a, a))
    print(calculate_fid(model, features, a, b))

if __name__ == '__main__':
    test()
