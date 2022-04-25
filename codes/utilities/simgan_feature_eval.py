import math
import numpy as np
import pandas as pd
from scipy import integrate
import numpy as np
from scipy.special import kl_div
from scipy.stats import ks_2samp, wasserstein_distance
import torch
from scipy.stats import ks_2samp, wasserstein_distance, ttest_ind


### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from codes.utilities.custom_logging import ezLogging


def calc_feature_distances(refiners, validation_data, device):
    '''
    TODO...get the source from the room

    Find the best refiner and discriminator from the list of refiners and discriminators using the feature distances.

        Parameters:
            refiners (list(torch.nn)): list of refiners
            validation_data (simganData): SimGAN dataset

        Returns:

    '''
    #N_samples = 100
    all_real = validation_data.real_raw.squeeze()#[:N_samples]
    all_simulated = validation_data.simulated_raw#[:N_samples]
    simulated_tensor = torch.tensor(all_simulated, dtype=torch.float, device=device)    

    # Calculate kl_div and wasserstein distance for features
    fe = FeatureExtractor()
    real_features = fe.get_features(all_real)
    feature_scores = {}
    for id_R, R in enumerate(refiners):
        refined_tensor = R(simulated_tensor.clone())
        refined = refined_tensor.detach().numpy().squeeze()
        refined_features = fe.get_features(refined)

        # Normalize features
        mins = np.expand_dims(np.min(np.concatenate([real_features, refined_features], axis=1), axis=1), axis=1)
        maxs = np.expand_dims(np.max(np.concatenate([real_features, refined_features], axis=1), axis=1), axis=1)
        normalized_real_features = (real_features - mins) / (maxs - mins)
        normalized_refined_features = (refined_features - mins) / (maxs - mins)

        kl_div, wasserstein_dist, ks_stat, pval = get_sampled_distribution_relation_scores(normalized_real_features.T, normalized_refined_features.T, bin=True)
        feature_scores[id_R] = {'kl_div': kl_div, 'wasserstein_dist': wasserstein_dist, 'ks_stat': ks_stat, 'sampled_pval': pval}

    mins = np.expand_dims(np.min(np.concatenate([real_features, real_features], axis=1), axis=1), axis=1)
    maxs = np.expand_dims(np.max(np.concatenate([real_features, real_features], axis=1), axis=1), axis=1)
    normalized_real_features = (real_features - mins) / (maxs - mins)
    feature_scores = pd.DataFrame.from_dict(feature_scores, orient='index')
    return feature_scores

def calc_t_tests(refiners, validation_data, device):
    '''
    Find the best refiner and discriminator from the list of refiners and discriminators using Welsh's t-tests.

        Parameters:
            refiners (list(torch.nn)): list of refiners
            validation_data (simganData): SimGAN dataset

        Returns:

    '''
    all_real = validation_data.real_raw.squeeze()
    all_simulated = validation_data.simulated_raw
    simulated_tensor = torch.tensor(all_simulated, dtype=torch.float, device=device)    

    # Calculate kl_div and wasserstein distance for features
    fe = FeatureExtractor()
    real_features = fe.get_features(all_real)
    feature_scores = {}
    for id_R, R in enumerate(refiners):
        refined_tensor = R(simulated_tensor.clone())
        refined = refined_tensor.detach().numpy().squeeze()
        refined_features = fe.get_features(refined)

        # Normalize features
        mins = np.expand_dims(np.min(np.concatenate([real_features, refined_features], axis=1), axis=1), axis=1)
        maxs = np.expand_dims(np.max(np.concatenate([real_features, refined_features], axis=1), axis=1), axis=1)
        normalized_real_features = (real_features - mins) / (maxs - mins)
        normalized_refined_features = (refined_features - mins) / (maxs - mins)

        condensed_wave_p_val = get_wave_t_test(normalized_real_features.T, normalized_refined_features.T)
        auc_p_val = get_auc_t_test(normalized_real_features.T, normalized_refined_features.T)
        kl_div, wasserstein_dist, ks_stat, pval = get_condensed_wave_dist(normalized_real_features.T, normalized_refined_features.T)
        avg_feat_p_val = get_multi_feature_average_t_test(normalized_real_features.T, normalized_refined_features.T)
        num_significant_feat = get_num_significant(normalized_real_features.T, normalized_refined_features.T, alpha=0.05)
        feature_scores[id_R] = {'condensed_kl_div': kl_div, 'condensed_wasserstein_dist': wasserstein_dist, 'condensed_ks_stat': ks_stat,\
            'condensed_ks_pval': pval, 'auc_pval': auc_p_val, 'condensed_wave_pval': condensed_wave_p_val, \
            'num_sig': num_significant_feat, 'avg_feat_pval': avg_feat_p_val}

    mins = np.expand_dims(np.min(np.concatenate([real_features, real_features], axis=1), axis=1), axis=1)
    maxs = np.expand_dims(np.max(np.concatenate([real_features, real_features], axis=1), axis=1), axis=1)
    normalized_real_features = (real_features - mins) / (maxs - mins)
    feature_scores = pd.DataFrame.from_dict(feature_scores, orient='index')
    return feature_scores

def estimated_trials(n, m): 
    '''
    For two unequally sized distributions, we want to randomly select a batch of m samples
    from both distributions and compare them. We want to continue sampling and comparing
    until we are confident that all n samples from the larger distributions have been sampled
    at least once. 

    This problem is an extension of the coupon collector problem
    https://en.wikipedia.org/wiki/Coupon_collector%27s_problem

    Specifically, we are looking into a Batched coupon collector problem
    Unfortunately this solution is mathematically impossible due to the size of n
    https://math.stackexchange.com/questions/3278200/iteratively-replacing-3-chocolates-in-a-box-of-10/3278285#3278285

    So we are running a simplified version by splitting the dataset into batches of samples,
    where we use n/m as the number of samples and assume m is 1. This means we are instead
    comparing across batches instead of samples, meaning we will be confident that each batch
    will be sampled at least once. 

    This function returns the expected number of times we need to sample the 
    distributions in order to see all n/m batches if we continue sampling batches of size m.

        Parameters:
            n (int): size of larger distribution
            m (int): batch size 

        Returns:
            expected_trials (int): number of trials needed to sample everything once
    '''
    num_batches = n//m
    euler_mascheroni = 0.5772156649
    expected_trials = num_batches * np.log(num_batches) + euler_mascheroni * num_batches + 0.5
    return int(expected_trials)

def divide_pad_distributions(dist1, dist2, batch_size):
    '''
    Divide the distributions to be evenly divisible by the batch size and pad the 
    distributions to fit the correct size. To pad the batches
    that are too small, we simply select random samples from the overall distribution.
        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): MxD numpy array 
            batch_size (int): given batch size of distribution


        Returns:
            padded_dist1 (ndarray), padded_dist2 (ndarray): two numpy arrays, each with batches of samples of size batch_size
    '''
    split_dist1 = np.array_split(dist1, math.ceil(len(dist1)/batch_size))
    split_dist2 = np.array_split(dist2, math.ceil(len(dist2)/batch_size))

    for i, batch in enumerate(split_dist1):
        if len(batch) != batch_size:
            random_samples = dist1[np.random.randint(dist1.shape[0], size=(batch_size - len(split_dist1[i]))), :]
            split_dist1[i] = np.vstack((split_dist1[i], random_samples))
    
    for j, batch in enumerate(split_dist2):
        if len(batch) != batch_size:
            random_samples = dist2[np.random.randint(dist2.shape[0], size=(batch_size - len(split_dist2[j]))), :]
            split_dist2[j] = np.vstack((split_dist2[j], random_samples))

    return split_dist1, split_dist2


def get_full_kl_div(dist1, dist2, batch_size, bin = True, clip_negatives=True, clip_lower_bound=0.0001):
    '''
    Calculate the kl divergence between the 2 distributions based on matching each batch with each other
        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): MxD numpy array 
            batch_size (int): given batch size of distribution
            clip_negatives (boolean): indicates whether we should clip the lower end of the disributions, can be useful to prevent infinite kl divergence
            clip_lower_bound (float): the lower bound to clip values at

        Returns:
            mean_kl_div (ndarray), median_kl_dv (ndarray): two numpy arrays, each with shape (D,), of the kl divergences
    '''

    if clip_negatives:
        # Clip values to positive to give reasonable values
        dist1 = np.clip(dist1, clip_lower_bound, dist1.max())
        dist2 = np.clip(dist2, clip_lower_bound, dist2.max())
    # Shuffle arrays
    np.random.shuffle(dist1)
    np.random.shuffle(dist2)

    dist1, dist2 = divide_pad_distributions(dist1, dist2, batch_size)

    #print("Number of combinations: ", len(dist1) * len(dist2))

    log_counter = 0
    kl_divs = None
    for real_batch in dist1:
        dist1_sample = real_batch
        np.random.shuffle(dist1_sample)
        for sim_batch in dist2:
            dist2_sample = sim_batch
            np.random.shuffle(dist2_sample)

            axis = 1 if bin else 0
            if kl_divs is not None:
                kl_divs = np.vstack((kl_divs, kl_div(dist1_sample, dist2_sample).sum(axis=axis)))
            else:
                kl_divs = kl_div(dist1_sample, dist2_sample).sum(axis=axis)

            #print("Combo: ", log_counter)
            log_counter += 1

    return np.mean(kl_divs), np.median(kl_divs)

def get_sampled_kl_div(dist1, dist2, batch_size, bin = True, clip_negatives=True, clip_lower_bound=0.0001):
    '''
    Calculate the kl divergence between the 2 distributions based on random sampling
        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): MxD numpy array 
            batch_size (int): given batch size of distribution
            clip_negatives (boolean): indicates whether we should clip the lower end of the disributions, can be useful to prevent infinite kl divergence
            clip_lower_bound (float): the lower bound to clip values at

        Returns:
            mean_kl_div (ndarray), median_kl_dv (ndarray): two numpy arrays, each with shape (D,), of the kl divergences
    '''

    larger_dist_size = max(len(dist1), len(dist2))
    num_trials = estimated_trials(larger_dist_size, batch_size)
    #print("Number of trials: ", num_trials)

    if clip_negatives:
        # Clip values to positive to give reasonable values
        dist1 = np.clip(dist1, clip_lower_bound, dist1.max())
        dist2 = np.clip(dist2, clip_lower_bound, dist2.max())
    # Shuffle arrays
    np.random.shuffle(dist1)
    np.random.shuffle(dist2)

    dist1, dist2 = divide_pad_distributions(dist1, dist2, batch_size)

    kl_divs = None
    for i in range(num_trials):
        #print("Trial started: ", i)
        dist1_sample = dist1[np.random.randint(len(dist1))]
        dist2_sample = dist2[np.random.randint(len(dist2))]
        np.random.shuffle(dist1_sample)
        np.random.shuffle(dist2_sample)

        axis = 1 if bin else 0
        if kl_divs is not None:
            kl_divs = np.vstack((kl_divs, kl_div(dist1_sample, dist2_sample).sum(axis=axis)))
        else:
            kl_divs = kl_div(dist1_sample, dist2_sample).sum(axis=axis)

    return np.mean(kl_divs), np.median(kl_divs)

def get_average_kl_div(dist1, dist2, bin=True, clip_negatives=True, clip_lower_bound=0.0001):
    '''
    Calculate the kl divergence between the 2 distributions
        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): NxD numpy array 
            clip_negatives (boolean): indicates whether we should clip the lower end of the disributions, can be useful to prevent infinite kl divergence
            clip_lower_bound (float): the lower bound to clip values at

        Returns:
            mean_kl_div (ndarray), median_kl_dv (ndarray): two numpy arrays, each with shape (D,), of the kl divergences
    '''
    if clip_negatives:
        # Clip values to positive to give reasonable values
        dist1 = np.clip(dist1, clip_lower_bound, dist1.max())
        dist2 = np.clip(dist2, clip_lower_bound, dist2.max())
    # Shuffle arrays
    np.random.shuffle(dist1)
    np.random.shuffle(dist2)

    axis = 1 if bin else 0
    kl_divs = kl_div(dist1, dist2).sum(axis=axis)
    return np.mean(kl_divs), np.median(kl_divs)

def get_full_wasserstein(dist1, dist2, batch_size, bin=True):
    '''
    Estimate the average wasserstein distance between the 2 distributions based on matching each batch with each other
        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): MxD numpy array 
            batch_size (int): given batch size of distribution

        Returns:
            mean_wasserstein_dist (ndarray), median_wasserstein_dist (ndarray): two numpy arrays, each with shape (D,), of the average wasserstein divergences
    '''
    # Shuffle arrays
    np.random.shuffle(dist1)
    np.random.shuffle(dist2)

    dist1, dist2 = divide_pad_distributions(dist1, dist2, batch_size)

    #print("Number of combinations: ", len(dist1) * len(dist2))

    log_counter = 0
    wassersteins = list()
    for real_batch in dist1:
        dist1_sample = real_batch
        np.random.shuffle(dist1_sample)
        for sim_batch in dist2:
            dist2_sample = sim_batch
            np.random.shuffle(dist2_sample)

            if bin:
                for j in range(len(dist1_sample[0])):
                    wassersteins.append(wasserstein_distance(dist1_sample[:, j], dist2_sample[:, j]))
            else:
                for j in range(batch_size):
                    # compute 1d wasserstein between 2 random signals
                    wassersteins.append(wasserstein_distance(dist1_sample[j].squeeze(), dist2_sample[j].squeeze()))
            
            #print("Combo: ", log_counter)
            log_counter += 1

    return np.mean(np.array(wassersteins)), np.median(np.array(wassersteins))

def get_sampled_wasserstein(dist1, dist2, batch_size, bin=True):
    '''
    Estimate the average wasserstein distance between the 2 distributions based on random sampling
        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): MxD numpy array 
            batch_size (int): given batch size of distribution

        Returns:
            mean_wasserstein_dist (ndarray), median_wasserstein_dist (ndarray): two numpy arrays, each with shape (D,), of the average wasserstein divergences
    '''

    larger_dist_size = max(len(dist1), len(dist2))
    num_trials = estimated_trials(larger_dist_size, batch_size)
    #print("Number of trials: ", num_trials)

    # Shuffle arrays
    np.random.shuffle(dist1)
    np.random.shuffle(dist2)

    dist1, dist2 = divide_pad_distributions(dist1, dist2, batch_size)

    wassersteins = list()
    for i in range(num_trials):
        #print("Trial started: ", i)
        dist1_sample = dist1[np.random.randint(len(dist1))]
        dist2_sample = dist2[np.random.randint(len(dist2))]
        np.random.shuffle(dist1_sample)
        np.random.shuffle(dist2_sample)

        if bin:
            for j in range(len(dist1_sample[0])):
                wassersteins.append(wasserstein_distance(dist1_sample[:, j], dist2_sample[:, j]))
        else:
            for j in range(batch_size):
                # compute 1d wasserstein between 2 random signals
                wassersteins.append(wasserstein_distance(dist1_sample[j].squeeze(), dist2_sample[j].squeeze()))

    return np.mean(np.array(wassersteins)), np.median(np.array(wassersteins))
    

def estimate_average_wasserstein(dist1, dist2, bin=True):
    '''
    Estimate the average wasserstein distance between the 2 distributions
        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): NxD numpy array 
            num_rounds (int): number of samples to draw and find avg. wasserstein difference between

        Returns:
            mean_wasserstein_dist (ndarray), median_wasserstein_dist (ndarray): two numpy arrays, each with shape (D,), of the average wasserstein divergences
    '''
    wassersteins = list()

    if bin:
        for j in range(len(dist1[0])):
            wassersteins.append(wasserstein_distance(dist1[:, j], dist2[:, j]))
    else:
        for j in range(len(dist1)):
            # compute 1d wasserstein between 2 random signals
            wassersteins.append(wasserstein_distance(dist1[j].squeeze(), dist2[j].squeeze()))


    return np.mean(wassersteins), np.median(wassersteins)

def get_full_ks_stat(dist1, dist2, batch_size, use_median=False, bin=True):
    '''
    Estimate the average ks-stat between the 2 distributions based on matching each batch with each other
        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): MxD numpy array 
            batch_size (int): given batch size of distribution

        Returns:
            mean_ks_stat (ndarray), mean_pvalue (ndarray): two numpy arrays, each with shape (D,), of the mean ks stats/pvalue
    '''
    # Shuffle arrays
    np.random.shuffle(dist1)
    np.random.shuffle(dist2)

    dist1, dist2 = divide_pad_distributions(dist1, dist2, batch_size)
    #print("Number of combinations: ", len(dist1) * len(dist2))

    log_counter = 0
    ks_stats = list()
    for real_batch in dist1:
        dist1_sample = real_batch
        np.random.shuffle(dist1_sample)
        for sim_batch in dist2:
            dist2_sample = sim_batch
            np.random.shuffle(dist2_sample)

            if bin:
                for j in range(len(dist1_sample[0])):
                    ks_stat = ks_2samp(dist1_sample[:, j], dist2_sample[:, j])
                    ks_stats.append([ks_stat.statistic, ks_stat.pvalue])
            else:
                for j in range(len(dist1_sample)):
                    ks_stat = ks_2samp(dist1_sample[j], dist2_sample[j])
                    ks_stats.append([ks_stat.statistic, ks_stat.pvalue])

            #print("Combo: ", log_counter)
            log_counter += 1

    ks_stats = np.array(ks_stats)
    if use_median:
        return np.median(ks_stats, axis=0)

    return ks_stats.mean(axis=0)

def get_sampled_ks_stat(dist1, dist2, batch_size, use_median=False, bin=True):
    '''
    Estimate the average ks-stat between the 2 distributions based on random sampling
        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): MxD numpy array 
            batch_size (int): given batch size of distribution

        Returns:
            mean_ks_stat (ndarray), mean_pvalue (ndarray): two numpy arrays, each with shape (D,), of the mean ks stats/pvalue
    '''
    larger_dist_size = max(len(dist1), len(dist2))
    num_trials = estimated_trials(larger_dist_size, batch_size)
    #print("Number of trials: ", num_trials)

    # Shuffle arrays
    np.random.shuffle(dist1)
    np.random.shuffle(dist2)

    dist1, dist2 = divide_pad_distributions(dist1, dist2, batch_size)

    ks_stats = list()
    for i in range(num_trials):
        #print("Trial started: ", i)
        dist1_sample = dist1[np.random.randint(len(dist1))]
        dist2_sample = dist2[np.random.randint(len(dist2))]
        np.random.shuffle(dist1_sample)
        np.random.shuffle(dist2_sample)

        if bin:
            for j in range(len(dist1_sample[0])):
                ks_stat = ks_2samp(dist1_sample[:, j], dist2_sample[:, j])
                ks_stats.append([ks_stat.statistic, ks_stat.pvalue])
        else:
            for j in range(len(dist1_sample)):
                ks_stat = ks_2samp(dist1_sample[j], dist2_sample[j])
                ks_stats.append([ks_stat.statistic, ks_stat.pvalue])

    ks_stats = np.array(ks_stats)
    if use_median:
        return np.median(ks_stats, axis=0)

    return ks_stats.mean(axis=0)

def get_average_ks_stat(dist1, dist2, use_median=False, bin=True):
    '''
    Estimate the average ks-stat between the 2 distributions
        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): NxD numpy array 

        Returns:
            mean_ks_stat (ndarray), mean_pvalue (ndarray): two numpy arrays, each with shape (D,), of the mean ks stats/pvalue
    '''
    ks_stats = []
    if bin:
        for j in range(len(dist1[0])):
            ks_stat = ks_2samp(dist1[:, j], dist2[:, j])
            ks_stats.append([ks_stat.statistic, ks_stat.pvalue])
    else:
        for j in range(len(dist1)):
            ks_stat = ks_2samp(dist1[j], dist2[j])
            ks_stats.append([ks_stat.statistic, ks_stat.pvalue])

    ks_stats = np.array(ks_stats)
    if use_median:
        np.median(ks_stats, axis=0)
    return ks_stats.mean(axis=0)

def get_distribution_relation_scores(dist1, dist2, bin, use_median=False, clip_for_kl=True):
    '''
    Calculate and return a set of scores that relate two distributions. Currently includes the average Wasserstein distance
    and average KL-Divergence between two random samples and the average Kolmogorov-Smirnov test value across feature values 
    from the two distributions.

        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist1 (ndarray): NxD numpy array 
            use_median: return median instead of mean statistics

        Returns:
            average_kl_div (ndarray), average_wasserstein_dist (ndarray), average_pvalue (ndarray): three numpy arrays, each with shape (D,), of the average kl-div, 
                wasserstein dist, and p value
    '''
    mean_kl_div, median_kl_div = get_average_kl_div(dist1, dist2, clip_negatives=clip_for_kl, bin=bin)
    mean_wasserstein_dist, median_wasserstein_dist = estimate_average_wasserstein(dist1, dist2, bin=bin)
    ks_stat, pvalue = get_average_ks_stat(dist1, dist2, use_median, bin=bin) #TODO: consider using the pvalue somehow
    if use_median:
        return median_kl_div, median_wasserstein_dist, ks_stat
    return mean_kl_div, mean_wasserstein_dist, ks_stat

def get_sampled_distribution_relation_scores(dist1, dist2, bin, use_median=False, clip_for_kl=True, batch_size=4):
    '''
    Calculate and return a set of scores that relate two distributions. Currently includes the average Wasserstein distance
    and average KL-Divergence between two random samples and the average Kolmogorov-Smirnov test value across feature values 
    from the two distributions.

        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): MxD numpy array 
            use_median: return median instead of mean statistics

        Returns:
            average_kl_div (ndarray), average_wasserstein_dist (ndarray), average_pvalue (ndarray): three numpy arrays, each with shape (D,), of the average kl-div, 
                wasserstein dist, and p value
    '''
    #print("Getting Sampled KL Div")
    #print("_____________________________________________")
    mean_kl_div, median_kl_div = get_sampled_kl_div(dist1, dist2, batch_size, bin=bin, clip_negatives=clip_for_kl)
    #print("Getting Sampled Wasserstein Dist")
    #print("_____________________________________________")
    mean_wasserstein_dist, median_wasserstein_dist = get_sampled_wasserstein(dist1, dist2, batch_size, bin=bin)
    #print("Getting Sampled KS Stat")
    #print("_____________________________________________")
    ks_stat, pvalue = get_sampled_ks_stat(dist1, dist2, batch_size, use_median, bin=bin) 
    if use_median:
        return median_kl_div, median_wasserstein_dist, ks_stat, pvalue
    return mean_kl_div, mean_wasserstein_dist, ks_stat, pvalue

def get_full_distribution_relation_scores(dist1, dist2, bin, use_median=False, clip_for_kl=True, batch_size=4):
    '''
    Calculate and return a set of scores that relate two distributions. Currently includes the average Wasserstein distance
    and average KL-Divergence between two random samples and the average Kolmogorov-Smirnov test value across feature values 
    from the two distributions.

        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): MxD numpy array 
            use_median: return median instead of mean statistics

        Returns:
            average_kl_div (ndarray), average_wasserstein_dist (ndarray), average_pvalue (ndarray): three numpy arrays, each with shape (D,), of the average kl-div, 
                wasserstein dist, and p value
    '''
    #print("Getting Full KL Div")
    #print("_____________________________________________")
    mean_kl_div, median_kl_div = get_full_kl_div(dist1, dist2, batch_size, clip_negatives=clip_for_kl, bin=bin)
    #print("Getting Full Wasserstein Dist")
    #print("_____________________________________________")
    mean_wasserstein_dist, median_wasserstein_dist = get_full_wasserstein(dist1, dist2, batch_size, bin=bin)
    #print("Getting Full KS Stat Dist")
    #print("_____________________________________________")
    ks_stat, pvalue = get_full_ks_stat(dist1, dist2, batch_size, use_median, bin=bin) 
    if use_median:
        return median_kl_div, median_wasserstein_dist, ks_stat, pvalue
    return mean_kl_div, mean_wasserstein_dist, ks_stat, pvalue

def get_wave_t_test(dist1, dist2, use_median=False):
    '''
    Create a mean/median waveform from both distributions then run a Welch's t-test to see
    if there is a significant difference between the resulting mean/median wave forms

        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): MxD numpy array 
            use_median: return median instead of mean statistics

        Returns:
            average_pvalue (float): average p value for the difference
    '''
    if use_median:
        x = np.median(dist1, axis=0)
        y = np.median(dist2, axis=0)
    else:
        x = np.mean(dist1, axis=0)
        y = np.mean(dist2, axis=0)
    _, p = ttest_ind(x, y, equal_var = False)

    return p

def get_auc_t_test(dist1, dist2):
    '''
    Calculate the area under the curves of the waveforms from both distributions 
    then run a Welch's t-test to see if there is a significant difference between them

        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): MxD numpy array 
            use_median: return median instead of mean statistics

        Returns:
            average_pvalue (float): average p value for the difference
    '''
    x = np.sum(dist1, axis=1)
    y = np.sum(dist2, axis=1)
    _, p = ttest_ind(x, y, equal_var = False)

    return p

def get_single_feature_t_test(dist1, dist2, index):
    '''
    Select a feature from both distributions then run a Welch's t-test 
    to see if there is a significant difference between them

        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): MxD numpy array 
            index: the feature to be selected

        Returns:
            pvalue (float): p value for the difference in feature distributions
    '''
    x = dist1[:, index]
    y = dist2[:, index]
    _, pvalue = ttest_ind(x, y, equal_var = False)

    return pvalue

def get_multi_feature_average_t_test(dist1, dist2):
    '''
    Select all features from both distributions then run a Welch's t-test
    on them individually to see if there is a significant difference between them
    then average the resultant p-values from each feature

        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): MxD numpy array 

        Returns:
            average_pvalue (float): average p value for the difference
    '''
    average_pvalue = 0
    num_features = dist1.shape[1]

    for index in range(num_features):
        x = dist1[:, index]
        y = dist2[:, index]
        _, p = ttest_ind(x, y, equal_var = False)
        average_pvalue += p

    average_pvalue /= num_features

    return average_pvalue

def get_num_significant(dist1, dist2, alpha):
    '''
    Select all features from both distributions then run a Welch's t-test
    on them individually to see if there is a significant difference between them
    then count how many features are statistically significant

        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): MxD numpy array 
            alpha (float): confidence threshold

        Returns:
            num_sig (int): number of features statistically significant
    '''
    num_sig = 0
    num_features = dist1.shape[1]

    for index in range(num_features):
        x = dist1[:, index]
        y = dist2[:, index]
        _, p = ttest_ind(x, y, equal_var = False)
        if p <= alpha:
            num_sig += 1

    return num_sig

def get_condensed_wave_dist(dist1, dist2, use_median=False, clip_for_kl=True):
    '''
    Create a mean/median waveform from both distributions then run a statistical distance
    metric to see how different the results are

        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist2 (ndarray): MxD numpy array 
            use_median: return median instead of mean statistics

        Returns:
            average_pvalue (float): average p value for the difference
    '''
    if use_median:
        x = np.median(dist1, axis=0)
        y = np.median(dist2, axis=0)
    else:
        x = np.mean(dist1, axis=0)
        y = np.mean(dist2, axis=0)

    x = np.expand_dims(x, axis=0)
    y = np.expand_dims(y, axis=0)
    mean_kl_div, median_kl_div = get_average_kl_div(x, y, clip_negatives=clip_for_kl, bin=False)
    wassersteins = wasserstein_distance(x[0], y[0])
    mean_wasserstein_dist, median_wasserstein_dist = np.mean(wassersteins), np.median(wassersteins)
    ks_stat, pvalue = get_average_ks_stat(x, y, use_median, bin=False) 
    if use_median:
        return median_kl_div, median_wasserstein_dist, ks_stat, pvalue
    return mean_kl_div, mean_wasserstein_dist, ks_stat, pvalue

class FeatureExtractor:
    def __init__(self, use_phdif=True, use_phratio=True, use_pdist=True, use_thdiff=True, use_thdist=True,
        use_area=True, use_changes=True, use_roughness=True, use_th=True):
        self.use_phdif = use_phdif
        self.use_phratio = use_phratio
        self.use_pdist = use_pdist
        self.use_thdiff = use_thdiff
        self.use_thdist = use_thdist
        self.use_area = use_area
        self.use_changes = use_changes
        self.use_roughness = use_roughness
        self.use_th = use_th
        self.feature_names = []

        self.num_features = 0
        if use_phdif:
            self.num_features += 1
            self.feature_names.append("phdif")
        if use_phratio:
            self.num_features += 1
            self.feature_names.append("phratio")
        if use_pdist:
            self.num_features += 1
            self.feature_names.append("pdist")
        if use_thdiff:
            self.num_features += 2
            self.feature_names.append("thdiff0")
            self.feature_names.append("thdiff1")
        if use_thdist:
            self.num_features += 2
            self.feature_names.append("thdist0")
            self.feature_names.append("thdist1")
        if use_area:
            self.num_features += 1
            self.feature_names.append("area")
        if use_changes:
            self.num_features += 1
            self.feature_names.append("changes")
        if use_roughness:
            self.num_features += 1
            self.feature_names.append("roughness")
        if use_th:
            self.num_features += 1
            self.feature_names.append("th")

    def get_features(self, dataset):
        '''
        Calculate the features for the datasets using the feature extractors in this file
            Parameters:
                dataset (ndarray): MxN numpy array with M N-dimensional waveforms to get features from

            Returns:
                features (ndarray): MxN' numpy array of N' features for M signals
                (Rodd Added) - LIES! it's actually N'xM so we have to transpose whatever gets returned
                    ...could be fixed if np.expand_dims axis is changed from 0 to 1 I think
        '''
        features = []
        if self.use_phdif:
            features.append(np.expand_dims(get_peak_height_difference(dataset), 0))
        if self.use_phratio:
            features.append(np.expand_dims(get_peak_height_ratio(dataset), 0))
        if self.use_pdist:
            features.append(np.expand_dims(get_peak_distance(dataset), 0))
        if self.use_thdiff:
            features.append(get_peaks_to_trough_height_difference(dataset))
        if self.use_thdist:
            features.append(get_peaks_to_trough_distance(dataset))
        if self.use_area:
            features.append(np.expand_dims(get_area(dataset), 0))
        if self.use_changes:
            features.append(np.expand_dims(get_changes(dataset), 0))
        if self.use_roughness:
            features.append(np.expand_dims(get_roughness(dataset), 0))
        if self.use_th:
            features.append(np.expand_dims(get_trough_height(dataset), 0))

        features = np.concatenate(features)
        return features

def get_peaks(data, start, end):
    '''
    Calculates the indices of the peaks
        Parameters:
            data (ndarray): MxN numpy array with M N-dimensional waveforms to calculate peaks with
            lrs (int): the index of the start of the peak's range
            lre (int): the index of the end of the peak's range

        Returns:
            peaks (ndarray): Mx1 numpy array of peak indices
    '''
    peak_range = data[:, start:end]
    return np.argmax(peak_range, axis=1) + start


def get_troughs(data, start, end):
    '''
    Calculates the indices of the troughs
        Parameters:
            data (ndarray): MxN numpy array with M N-dimensional waveforms to calculate troughs with
            lrs (int): the index of the start of the trough's range
            lre (int): the index of the end of the trough's range

        Returns:
            troughs (ndarray): Mx1 numpy array of trough indices
    '''
    trough_range = data[:, start:end]
    return np.argmin(trough_range, axis=1) + start


def get_peak_distance(data, lrs=10, lre=40, rrs=40, rre=80):
    '''
    Calculates the distance between two peaks in an array of waveforms
        Parameters:
            data (ndarray): MxN numpy array with M N-dimensional waveforms to calculate peak distances with
            lrs (int): the index of the start of the left peak's range
            lre (int): the index of the end of the left peak's range
            rrs (int): the index of the start of the right peak's range
            rre (int): the index of the end of the right peak's range

        Returns:
            distance (ndarray): numpy array of the distances between the left and right peaks
    '''
    left_peak_inds = get_peaks(data, lrs, lre)
    right_peak_inds = get_peaks(data, rrs, rre)
    distances = right_peak_inds - left_peak_inds
    return distances


def get_peak_height_difference(data, lrs=10, lre=40, rrs=40, rre=80):
    '''
    Calculates the height difference between two peaks in an array of waveforms
        Parameters:
            data (ndarray): MxN numpy array with M N-dimensional waveforms to calculate peak height differences with
            lrs (int): the index of the start of the left peak's range
            lre (int): the index of the end of the left peak's range
            rrs (int): the index of the start of the right peak's range
            rre (int): the index of the end of the right peak's range

        Returns:
            distance (ndarray): numpy array of the height differences between the left and right peaks
    '''
    left_peak_inds = get_peaks(data, lrs, lre)
    right_peak_inds = get_peaks(data, rrs, rre)
    left_peaks = data[list(range(len(left_peak_inds))), left_peak_inds]
    right_peaks = data[list(range(len(right_peak_inds))), right_peak_inds]
    return left_peaks - right_peaks


def get_peak_height_ratio(data, lrs=10, lre=40, rrs=40, rre=80):
    '''
    Calculates the height ratios between two peaks in an array of waveforms
        Parameters:
            data (ndarray): MxN numpy array with M N-dimensional waveforms to calculate peak height ratios with
            lrs (int): the index of the start of the left peak's range
            lre (int): the index of the end of the left peak's range
            rrs (int): the index of the start of the right peak's range
            rre (int): the index of the end of the right peak's range

        Returns:
            distance (ndarray): numpy array of the height differences between the left and right peaks
    '''
    left_peak_inds = get_peaks(data, lrs, lre)
    right_peak_inds = get_peaks(data, rrs, rre)
    left_peaks = data[list(range(len(left_peak_inds))), left_peak_inds]
    right_peaks = data[list(range(len(right_peak_inds))), right_peak_inds]
    return left_peaks / right_peaks

def get_peaks_to_trough_height_difference(data, lrs=10, lre=40, rrs=40, rre=80):
    '''
    Calculates the height difference between the peaks and the trough
        Parameters:
            data (ndarray): MxN numpy array with M N-dimensional waveforms to calculate height difference
            lrs (int): the index of the start of the left peak's range
            lre (int): the index of the end of the left peak's range
            rrs (int): the index of the start of the right peak's range
            rre (int): the index of the end of the right peak's range

        Returns:
            distance (ndarray): 2xM numpy array of the height differences between the left and right peaks and the trough
    '''
    left_peak_inds = get_peaks(data, lrs, lre)
    right_peak_inds = get_peaks(data, rrs, rre)
    left_peaks = data[list(range(len(left_peak_inds))), left_peak_inds]
    right_peaks = data[list(range(len(right_peak_inds))), right_peak_inds]
    trough_inds = get_troughs(data, int((lre+lrs) / 2), int((rre+rrs) / 2))
    troughs = data[list(range(len(trough_inds))), trough_inds]
    return np.array([left_peaks - troughs, right_peaks - troughs])


def get_peaks_to_trough_distance(data, lrs=10, lre=40, rrs=40, rre=80):
    '''
    Calculates the distance between the peaks and the trough
        Parameters:
            data (ndarray): MxN numpy array with M N-dimensional waveforms to calculate the distances
            lrs (int): the index of the start of the left peak's range
            lre (int): the index of the end of the left peak's range
            rrs (int): the index of the start of the right peak's range
            rre (int): the index of the end of the right peak's range

        Returns:
            distance (ndarray): 2xM numpy array of the distances between the left and right peaks and the trough
    '''
    left_peak_inds = get_peaks(data, lrs, lre)
    right_peak_inds = get_peaks(data, rrs, rre)
    trough_inds = get_troughs(data, int((lre+lrs) / 2), int((rre+rrs) / 2))
    return np.array([trough_inds - left_peak_inds, right_peak_inds - trough_inds])


def get_mean_deviation_from_mean_signal(data):
    '''
    Calculates the mean deviation of an array of signals from the mean signal
        Parameters:
            data (ndarray): MxN numpy array with M N-dimensional waveforms

        Returns:
            mean_deviation_from_mean_signal (ndarray): 1xM numpy array of the mean deviation from mean signal
    '''
    mean_signal = np.mean(data, axis=0)
    deviation_from_mean_signal = np.abs(data - mean_signal)
    return np.mean(deviation_from_mean_signal, axis=0)

def get_mean_deviation_from_median_signal(data):
    '''
    Calculates the mean deviation of an array of signals from the median signal
        Parameters:
            data (ndarray): MxN numpy array with M N-dimensional waveforms

        Returns:
            mean_deviation_from_median_signal (ndarray): 1xM numpy array of the mean deviation from median signal
    '''
    median_signal = np.median(data, axis=0)
    deviation_from_median_signal = np.abs(data - median_signal)
    return np.mean(deviation_from_median_signal, axis=0)


def get_number_of_higher_right_peaks(data, lrs=10, lre=40, rrs=40, rre=80):
    '''
    Calculates the number of signals with a right peak that is higher than its left peak
        Parameters:
            data (ndarray): MxN numpy array with M N-dimensional waveforms
            lrs (int): the index of the start of the left peak's range
            lre (int): the index of the end of the left peak's range
            rrs (int): the index of the start of the right peak's range
            rre (int): the index of the end of the right peak's range

        Returns:
            n_signals (int): The number of signals
    '''
    left_peak_inds = get_peaks(data, lrs, lre)
    right_peak_inds = get_peaks(data, rrs, rre)
    left_peaks = data[list(range(len(left_peak_inds))), left_peak_inds]
    right_peaks = data[list(range(len(right_peak_inds))), right_peak_inds]
    return len(np.where(right_peaks > left_peaks)[0])


def get_area(data):
    '''
    Calculates the area under the signal's curve
        Parameters:
            data (ndarray): MxN numpy array with M N-dimensional waveforms to calculate the distances

        Returns:
            area (ndarray): 1xM numpy array of the area under the signal's curve
    '''
    area = np.zeros(data.shape[0])
    area = integrate.simps(data)
    return area


def get_changes(data):
    '''
    Calculates how many times the signal changes direction
        Parameters:
            data (ndarray): MxN numpy array with M N-dimensional waveforms to calculate the distances

        Returns:
            changes (ndarray): 1xM numpy array of the number of changes for the signal
    '''
    changes = np.zeros(data.shape[0])

    upVar = data[:, 1:] > data[:, 0:-1]
    downVar = data[:, 1:] < data[:, 0:-1]

    upwardsTrend = np.insert(upVar[:,:-1], 0, 0, axis=1)
    downwardsTrend = np.insert(downVar[:,:-1], 0, 0, axis=1)

    upChanges = upVar & downwardsTrend
    downChanges = downVar & upwardsTrend

    changes = np.sum(upChanges | downChanges, axis=1)

    return changes


def get_roughness(data):
    '''
    Calculates the roughness by [sum(rolling mean - actual)] / N. Higher score is less smooth
        Parameters:
            data (ndarray): MxN numpy array with M N-dimensional waveforms

        Returns:
            roughness (ndarray): 1xM numpy array of the roughness of each signal
    '''
    smoothed = np.copy(data)
    j = 0
    for j in np.arange(2, data.shape[1]-2):
        smoothed[:, j] = np.average(data[:, j-2:j+3], axis=1)
    dif = np.abs(smoothed - data)
    return np.sum(dif, axis=1) / data.shape[1]


def get_trough_height(data, lrs=10, lre=40, rrs=40, rre=80):
    '''
    Calculates the trough height
        Parameters:
            data (ndarray): MxN numpy array with M N-dimensional waveforms

        Returns:
            roughness (ndarray): 1xM numpy array of the height of each signal's trough
    '''
    trough_inds = get_troughs(data, int((lre + lrs) / 2), int((rre + rrs) / 2))
    trough_heights = data[list(range(len(data))), trough_inds]
    # print(trough_heights.shape)
    # print(trough_heights)
    # i = 0
    # for row in data:
    #     trough_heights[i] = row[trough_inds[i]]
    #     i += 1
    return trough_heights
