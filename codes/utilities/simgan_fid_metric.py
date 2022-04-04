# Frechet Inception Distance metric to be used for simgan performance  

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
from torchvision import transforms
from numpy.random import random
from scipy.linalg import sqrtm


features = {}
model = torch.hub.load('pytorch/vision:v0.8.0', 'inception_v3', pretrained=True)
#model = nn.Sequential(*list(model.children())[:-2])

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

model.avgpool.register_forward_hook(get_features('feats'))
model.eval()

def process(input, device):
    batch_size = input.shape[0]
    z = torch.zeros((batch_size, 23), device=device)
    h = torch.hstack([input, input, input, z])
    
    output = torch.stack([h for _ in range(299)])
    output = output.permute(1,0,2)
    
    output = torch.stack([output, output, output])
    output = output.permute(1,0,2,3)
    return output

def get_activations(model, inputs, device):    
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')
        model.to('cuda')

    inputs = process(inputs, device)

    with torch.no_grad():
        output = model(inputs)
        return features['feats'].squeeze()

#adapted from https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/ 
def calculate_fid(act1, act2):
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
    return fid

def calc_fid(ref, real, device):
    act1 = get_activations(model, ref, device).cpu().data
    act2 = get_activations(model, real, device).cpu().data
    return calculate_fid(act1, act2).item()

def get_fid_scores(refiners, validation_data, device):
    all_real = validation_data.real_raw
    all_sim = validation_data.simulated_raw
    
    batch_size = min([all_real.shape[0], all_sim.shape[0]])
   
    chosen_sim = torch.tensor(all_sim[:batch_size, :, :], dtype=torch.float, device=device)
    chosen_real = torch.tensor(all_real[:batch_size, :, :], dtype=torch.float, device=device)


    fid_scores = []
    for i, R in enumerate(refiners):
        ref = R(chosen_sim)
        fid = calc_fid(ref.squeeze(), chosen_real.squeeze(), device)
        fid_scores.append(fid)

    return fid_scores



def test():
    a = torch.ones([128, 92])
    b = torch.randn([128, 92])
    print(calc_fid(a,a))
    print(calc_fid(a,b))

if __name__ == '__main__':
    test()
