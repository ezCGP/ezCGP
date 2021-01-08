import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class dataset(Dataset):
    """
    Defines a torch Dataset used for training 
    """
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

class SimganFakeDataset():
    """
    Holds a simulated and real dataset, each composed of waves meant to mimic the ones in the room.
    The real dataset has an attenuated second and has some slightly different generation parameters.
    """
    def __init__(self, real_size=128**2, sim_size=256, batch_size=128):
        ### Get the real and simulated datasets
        # Can configure these, but they are a bit obtuse
        self.real_raw = self.gen_fake_dataset(real_size, [24,56], [0.5,0.5], [5,8], 
                                              [0.5,0.5], [1,0.64], [0.1,0.2],
                                              [[1,4,6],[5,8,10]], [0.25,0.25], True)
        self.simulated_raw = self.gen_fake_dataset(sim_size, [23.5,55.5], [0.5,0.5], [5,10],
                                                   [0.5,0.5], [1,0.78], [0.05,0.17],
                                                   [[4],[10]], [0.25,0.25], False)
        ### Get the labels
        self.labels_real = np.zeros(self.real_raw.shape[0])
        self.labels_simulated = np.zeros(self.simulated_raw.shape[0])

        ### Put the data into pytorch friendly format
        self.real = dataset(self.real_raw, self.labels_real)
        self.simulated = dataset(self.simulated_raw, self.labels_simulated)
        # kwargs = {'num_workers': 1, 'pin_memory': False} if cuda else {} # Put in the original code by Alex, but I'm not sure for what, commenting them out in the meantime
        self.real_loader = DataLoader(
            self.real,
            batch_size,
            shuffle=True,
            # **kwargs
        )
        self.simulated_loader = DataLoader(
            self.simulated,
            batch_size,
            shuffle=True,
            # **kwargs
        )
    
    def gen_fake_dataset(self, n_signals, peak_locs_mu, peak_locs_sigma, mu_sigma, avg_peak_std_dev, amp_mu, amp_sigma, freq_mu, freq_sigma, add_noise=False):
        """
        Code to generate our fake dataset that mocks the in-room 
        """
        data = []
        peaks = np.zeros((len(peak_locs_mu), n_signals))
        sigmas = np.zeros((len(peak_locs_sigma), n_signals))
        g_amplitudes = np.zeros((len(amp_mu), n_signals))
        freqs = []

        for i in range(0, len(peak_locs_mu)):
            peaks[i,:] = np.random.normal(loc = peak_locs_mu[i], scale = peak_locs_sigma[i], size = n_signals)
            sigmas[i,:] = np.random.normal(loc = mu_sigma[i], scale = peak_locs_sigma[i], size = n_signals)
            g_amplitudes[i,:] = np.random.normal(loc = amp_mu[i], scale = amp_sigma[i], size = n_signals)
        
        temp_f = []
        for i in range(0, len(freq_mu)):
            temp_f.append(np.array([np.random.normal(loc = m, scale = s, size = n_signals) for m, s in zip(freq_mu[i], freq_sigma)]))
        
        temp_f = np.array(temp_f)
        for i in range(0, n_signals):
            freqs.append(np.array(temp_f)[:,:,i].tolist())

        for i in range(0, n_signals):
            primary, residual = self.gen_siganl(peaks[:,i], sigmas[:,i], g_amplitudes[:,i], freqs[i])
            data.append((primary + residual).reshape((1,92)))

        data = np.array(data)

        if add_noise:
            # TODO: vectorize this code
            # Add random noise to the real signals scaled by value
            for i in range(len(data)):
                for j in range(len(data[i][0])):
                    data[i][0][j] += np.random.normal(0, .05) * np.clip(data[i][0][j], 0.5, 1)
            # Normalize
            for i in range(len(data)):
                data[i] -= data[i].min(axis=1)
                data[i] /= data[i].max(axis=1)
        
        return data
    
    def gen_siganl(self, peaks, sigmas, g_amplitudes, freqs, f_amplitudes = 0.02, fs = 1/60, size = 92):
        '''
        Generate a synthetic signal with sinusoidal components added at the peaks. 
        
        Inputs peaks, sigmas, g_amplitudes, and freqs to be list where each entry corresponds to the 
        first, second, ... etc gaussian components in the signal.

        Freqs entries can contain multiple frequencies in an array.
        '''
        signal = np.zeros(size)
        f_signal = np.zeros(size)
        for peak, sigma, amplitude, freq in zip(peaks, sigmas, g_amplitudes, freqs):
            temp = self.gen_gaussian_kernel(sigma, peak)
            signal += amplitude * ( temp / temp.max())
            x = np.linspace(0, len(signal)*fs, num = size)
            x[:(int(np.floor(peak-sigma)))] = 0
            x[(int(np.ceil(peak+sigma))):] = 0

            smoothing = temp / temp.max()
            for f in freq:
                f_signal += (f_amplitudes * np.sin(x*2*np.pi*f)) 

        return signal, f_signal

    def gen_gaussian_kernel(self, sigma, mu, size = 92):
        '''
        Generate a gaussian kernel of the desired size
        '''
        nstds = 1  
        xmax = 91 
        xmin = 0
        x =  np.linspace(xmin, xmax, size)

        kernel = ( (1 / (np.sqrt(2 * np.pi) * sigma**2)) * np.exp(-.5 * (((x - mu) ** 2) / sigma**2)) )

        return kernel
