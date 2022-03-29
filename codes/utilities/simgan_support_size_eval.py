import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
from scipy.special import kl_div
import torch
import pandas as pd

# TODO: speedup
def get_cross_correlation_similarities(samples):
    '''
    Calculate the cross correlation similarity for each pair of samples
        Parameters:
            samples (ndarray): MxN numpy array with M N-dim signals

        Returns:
            similarities (ndarray): MxM numpy array of the cross correlation similarities
    '''
    n_samples = samples.shape[0]
    similarities = np.zeros((n_samples, n_samples))
    # Get diagonal of matrix of similarities, as we don't want to include duplicate pairings
    for i in range(n_samples):
        for j in range(n_samples):
            # At (i,i), similarities will be maxed, but we don't want to consider them so we set the value to nan
            if j >= i:
                similarities[i, j] = np.nan
            else:
                similarities[i, j] = np.max(correlate(samples[i], samples[j]))
    return similarities


# TODO: speedup
def get_kl_divergences(samples):
    '''
    Calculate the kl divergence for each pair of samples
        Parameters:
            samples (ndarray): MxN numpy array with M N-dim signals

        Returns:
            divergences (ndarray): MxM numpy array of the kl divergences
    '''
    n_samples = samples.shape[0]
    divergences = np.zeros((n_samples, n_samples))
    # Get diagonal of matrix of similarities, as we don't want to include duplicate pairings
    for i in range(n_samples):
        for j in range(n_samples):
            # At (i,i), similarities will be maxed, but we don't want to consider them so we set the value to nan
            if j >= i:
                divergences[i, j] = np.inf
            else:
                divergences[i, j] = np.sum(kl_div(samples[i], samples[j]))
    return divergences


def get_mae_divergences(samples):
    '''
    Calculate the mae divergence for each pair of samples
        Parameters:
            samples (ndarray): MxN numpy array with M N-dim signals

        Returns:
            divergences (ndarray): MxM numpy array of the mae divergences
    '''
    n_samples = samples.shape[0]
    divergences = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
    # Get diagonal of matrix of divergences, as we don't want to include duplicate pairings
        divergences[i, i:] = np.mean(np.abs(samples[i:] - samples[i]), axis=1)
        # At (i,i), divergences are 0, but we don't want to consider them so we set the value to inf
        divergences[i, i] = np.inf
        # Also don't want to consider the bottom half of the matrix
        divergences[i, :i] = np.inf
    return divergences


def graph_top_n_similar_signals(samples, dir_path, top_n=20, method="mae"):
    '''
    Graph the top_n signal pairs by similarity side by side using the similarity/divergence method indicated
        Parameters:
            samples (ndarray): MxN numpy array with M N-dim signals
            dir_path (string): Path to the directory where the plots should be saved
            top_n (int): Number of similar signal pairs to graph
            method (str): The similarity/divergence method to use, possible values: 'cc', 'kl', 'mae'

    '''
    print(samples.shape) # N x 92

    # Start by getting the top_n closest signals using the similarity/divergence method indicated
    closest_inds = None
    metric_vals = None
    n_samples = samples.shape[0]
    if method =='cc':
        print('Using cross correlation')
        metric_vals = get_cross_correlation_similarities(samples)
        closest_inds = np.argpartition(-metric_vals.flatten(), top_n)[:top_n]
    elif method == 'kl':
        print('using kl divergence')
        metric_vals = get_kl_divergences(samples)
        closest_inds = np.argpartition(metric_vals.flatten(), top_n)[:top_n]
    elif method == 'mae':
        print('using mean absolute error')
        metric_vals = get_mae_divergences(samples)
        closest_inds = np.argpartition(metric_vals.flatten(), top_n)[:top_n]
    else:
        raise Exception('Invalid method for BPT: "{}", expected "cc", "kl", or "mae".'.format(method))
    
    # Transpose for easier graphing
    closest_inds = np.transpose(np.array([closest_inds // n_samples, closest_inds % n_samples]), (1, 0))

    # Graph each signal pair
    x = list(range(samples.shape[-1]))
    for i, index in enumerate(closest_inds):
        plt.plot(x, samples[index[0]])
        plt.plot(x, samples[index[1]])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Top ' + str(top_n) + ' closest signals - ' + method + ': ' + str(metric_vals[index[0], index[1]]))
        plt.savefig(dir_path + method + '_top_n_closest_' + str(i) + '.png')
        plt.cla()


def birthday_paradox_test(samples, same_signal_thresh, method="mae"):
    '''
    Calculate whether a match exists in the samples using the similarity/divergence method indicated
        Parameters:
            samples (ndarray): MxN numpy array with M N-dim signals
            same_signal_thresh (float): The threshold for a given similarity/divergence method that indicates two samples are the same 
            method (str): The similarity/divergence method to use, possible values: 'cc', 'kl', 'mae'

        Returns:
            (match_found, n_matches) (bool, int): A tuple with a boolean indicating whether a match was found and an int for the number of matches found
    '''
    # Start by getting the similarities or divergences depending on the method indicated
    metric_vals = None
    if method =='cc':
        metric_vals = get_cross_correlation_similarities(samples)
    elif method == 'kl':
        metric_vals = get_kl_divergences(samples)
    elif method == 'mae':
        metric_vals = get_mae_divergences(samples)
    else:
        raise Exception('Invalid method for BPT: "{}", expected "cc", "kl", or "mae".'.format(method))
    
    # Find where there are signal duplicates using the threshold
    same_signal_inds = None
    if method == 'cc':
        same_signal_inds = np.where(metric_vals > same_signal_thresh)
    else:
        same_signal_inds = np.where(metric_vals < same_signal_thresh)

    matching_signals = metric_vals[same_signal_inds]
    n_matches = len(matching_signals)
    return n_matches > 0, n_matches


def calculate_support_size(samples, same_signal_thresh, data_config=None, gen_func_kwargs=None, method="mae", iterations=100, starting_n_samples=100, max_samples=5000):
    '''
    Calculate the support size of the generating function using an iterative method that hones in on the support size using the birthday paradox test
        Parameters:
            samples (tensor): samples for testing
            same_signal_thresh (float): The threshold for a given similarity/divergence method that indicates two samples are the same 
            data_config (ET.Element): An ET xml element holding the data configuration needed for the gen_func
            gen_func_kwargs (dict): A dict of kwargs to pass into the gen_func
            method (str): The similarity/divergence method to use, possible values: 'cc', 'kl', 'mae'
            iterations (int): The number of iterations to check for matches in the generated samples
            starting_n_samples (int): The number of samples to initialize with, affects how long convergence takes
            max_samples (int): The maximum number of samples to check before giving up (runtime grows polynomially with # samples, so we have to have a cutoff)

        Returns:
            (support_size, percent iterations with duplicates) (int, float): A tuple with the int support size and a float representing the percent of duplicates 
                in the final iteration
    '''
    percent_duplicates = None
    n_samples = starting_n_samples
    while n_samples < max_samples:
        
        if n_samples == 0:
            return max_samples**2, percent_duplicates

        # Find percent of iterations with duplicates
        iterations_with_duplicates = 0
        for i in range(iterations):
            duplicates_found, _ = birthday_paradox_test(samples, same_signal_thresh, method)
            if duplicates_found:
                iterations_with_duplicates += 1
        
        # Determine whether the correct support_size has been found
        percent_duplicates = 100.0 * (1.0 * iterations_with_duplicates / iterations)
        print('N Samples: ', n_samples, ', Percent duplicates: ', percent_duplicates)
        if percent_duplicates < 40:
            n_samples = int(np.floor(n_samples * 1.5))
        elif percent_duplicates > 60:
            n_samples = int(np.floor(n_samples * 0.8))
        else:
            return n_samples**2, percent_duplicates

    return max_samples**2, percent_duplicates

def get_support_size(refiners, validation_data, device):
    '''
    Find the best refiner and discriminator from the list of refiners and discriminators using support size.

        Parameters:
            refiners (list(torch.nn)): list of refiners
            validation_data (simganData): SimGAN dataset

        Returns:

    '''
    all_simulated = validation_data.simulated_raw
    simulated_tensor = torch.tensor(all_simulated, dtype=torch.float, device=device) 
    support_sizes = {}
    for id_R, R in enumerate(refiners):
        refined = R(simulated_tensor).detach().cpu().numpy().squeeze() 
        support_size, percent_duplicates = calculate_support_size(refined, 0.012, method="mae")
        support_sizes[id_R] = support_size
    support_size_df = pd.DataFrame.from_dict(support_sizes, orient='index')
    return support_size_df