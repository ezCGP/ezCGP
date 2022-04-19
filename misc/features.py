'''
copy/paste from the OG simgan repo
'''
import numpy as np
from scipy import integrate
from scipy.signal import correlate
   
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
            roughness (ndarray): 1xM numpy array of the height of each signal's trough
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