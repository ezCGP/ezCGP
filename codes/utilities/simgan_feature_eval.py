
import numpy as np
import pandas as pd
from scipy import integrate
import numpy as np
from scipy.special import kl_div
from scipy.stats import ks_2samp, wasserstein_distance
import torch

def calc_feature_distances(refiners,
                           validation_data,
                           device):
    '''
    TODO...get the source from the room

    Find the best refiner and discriminator from the list of refiners and discriminators using the feature distances.

        Parameters:
            refiners (list(torch.nn)): list of refiners
            validation_data (simganData): SimGAN dataset

        Returns:

    '''
    N_samples = 100
    all_real = validation_data.real_raw.squeeze()[:N_samples]
    all_simulated = validation_data.simulated_raw[:N_samples]
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

        kl_div, wasserstein_dist, ks_stat = get_distribution_relation_scores(normalized_real_features.T, normalized_refined_features.T, clip_for_kl=False)
        feature_scores[id_R] = {'kl_div': kl_div, 'wasserstein_dist': wasserstein_dist, 'ks_stat': ks_stat}

    mins = np.expand_dims(np.min(np.concatenate([real_features, real_features], axis=1), axis=1), axis=1)
    maxs = np.expand_dims(np.max(np.concatenate([real_features, real_features], axis=1), axis=1), axis=1)
    normalized_real_features = (real_features - mins) / (maxs - mins)
    feature_scores = pd.DataFrame.from_dict(feature_scores, orient='index')
    return feature_scores.T

def calc_raw_waveform_distances(refiners,
                                validation_data,
                                device):
    '''
    TODO...get the source from the room

    Find the best refiner and discriminator from the list of refiners and discriminators using the feature distances.

        Parameters:
            refiners (list(torch.nn)): list of refiners
            validation_data (simganData): SimGAN dataset

        Returns:

    '''
    N_samples = 100
    all_real = validation_data.real_raw.squeeze()[:N_samples]
    all_simulated = validation_data.simulated_raw[:N_samples]
    simulated_tensor = torch.tensor(all_simulated, dtype=torch.float, device=device)

    # Calculate average kl_div and wasserstein distance for two random signals from the distributions
    raw_waveform_scores = {}
    for id_R, R in enumerate(refiners):
        refined_tensor = R(all_simulated.detach().clone())
        refined = refined_tensor.cpu().detach().numpy().squeeze()
        kl_div = get_distribution_relation_scores(all_real, refined)
        raw_waveform_scores[id_R] = {'kl_div': kl_div, 'wasserstein_dist': wasserstein_dist, 'ks_stat': ks_stat}

    raw_waveform_scores = pd.DataFrame.from_dict(raw_waveform_scores, orient='index')
    return raw_waveform_scores

def get_average_kl_div(dist1, dist2, clip_negatives=True, clip_lower_bound=0.0001):
    '''
    Calculate the kl divergence between the 2 distributions
        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist1 (ndarray): NxD numpy array 
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

    kl_divs = kl_div(dist1, dist2).sum(axis=1)
    return np.mean(kl_divs), np.median(kl_divs)

def estimate_average_wasserstein(dist1, dist2, num_rounds=100):
    '''
    Estimate the average wasserstein distance between the 2 distributions
        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist1 (ndarray): NxD numpy array 
            num_rounds (int): number of samples to draw and find avg. wasserstein difference between

        Returns:
            mean_wasserstein_dist (ndarray), median_wasserstein_dist (ndarray): two numpy arrays, each with shape (D,), of the average wasserstein divergences
    '''
    wassersteins = np.zeros((dist1.shape[0],))
    inds = np.arange(dist1.shape[0])
    for i in range(num_rounds):
        # compute 1d wasserstein between 2 random signals
        wassersteins[i] = wasserstein_distance(dist1[np.random.choice(inds, 1), :].squeeze(), dist2[np.random.choice(inds, 1), :].squeeze())
    
    return np.mean(wassersteins), np.median(wassersteins)

def get_average_ks_stat(dist1, dist2, use_median=True):
    '''
    Estimate the average ks-stat between the 2 distributions
        Parameters:
            dist1 (ndarray): NxD numpy array 
            dist1 (ndarray): NxD numpy array 

        Returns:
            mean_ks_stat (ndarray), mean_pvalue (ndarray): two numpy arrays, each with shape (D,), of the mean ks stats/pvalue
    '''
    ks_stats = []
    for i in range(dist1.shape[-1]):
        ks_stat = ks_2samp(dist1[:, i], dist2[:, i])
        ks_stats.append([ks_stat.statistic, ks_stat.pvalue])
    ks_stats = np.array(ks_stats)
    if use_median:
        np.median(ks_stats, axis=0)
    return ks_stats.mean(axis=0)

def get_distribution_relation_scores(dist1, dist2, use_median=True, clip_for_kl=True):
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
    mean_kl_div, median_kl_div = get_average_kl_div(dist1, dist2, clip_negatives=clip_for_kl)
    mean_wasserstein_dist, median_wasserstein_dist = estimate_average_wasserstein(dist1, dist2)
    ks_stat, pvalue = get_average_ks_stat(dist1, dist2, use_median) #TODO: consider using the pvalue somehow
    if use_median:
        return median_kl_div, median_wasserstein_dist, ks_stat
    return mean_kl_div, mean_wasserstein_dist, ks_stat

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

        self.num_features = 0
        if use_phdif:
            self.num_features += 1
        if use_phratio:
            self.num_features += 1
        if use_pdist:
            self.num_features += 1
        if use_thdiff:
            self.num_features += 2
        if use_thdist:
            self.num_features += 2
        if use_area:
            self.num_features += 1
        if use_changes:
            self.num_features += 1
        if use_roughness:
            self.num_features += 1
        if use_th:
            self.num_features += 1

    def get_features(self, dataset):
        '''
        Calculate the features for the datasets using the feature extractors in this file
            Parameters:
                dataset (ndarray): MxN numpy array with M N-dimensional waveforms to get features from

            Returns:
                features (ndarray): MxN' numpy array of N' features for M signals
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
