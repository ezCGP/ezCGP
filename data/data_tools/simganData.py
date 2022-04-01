import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class dataset(Dataset):
    """
    Defines a torch Dataset used for training 
    """
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

class DataHistoryBuffer():
    """
    Holds a buffer of data alreaady seen by the discriminator. This is used for training on previously seen data and is outlined in the original
    SimGAN paper. Read that for more details
    """
    def __init__(self, shape, max_size, batch_size):
        """
        Initialize the class's state.

        :param shape: Shape of the data to be stored in the history buffer
                    (e.g. (signal_length,)).
        :param max_size: Maximum number of data point that can be stored in the history buffer.
        :param batch_size: Batch size used to train GAN.
        """
        self.history_buffer = np.zeros((0, *shape)) # We use 0 in the first axis because we will be appending until we reach the max size
        self.max_size = max_size
        self.batch_size = batch_size

    def add(self, data_points, nb_to_add=None):
        """
        To be called during training of GAN. By default add batch_size // 2 data points to the history buffer each
        time the generator generates a new batch of data points.

        :param data_points: Array of data_points (usually a batch) to be added to the data history buffer.
        :param nb_to_add: The number of data points from `data_points` to add to the data history buffer
                        (batch_size / 2 by default).
        """
        if not nb_to_add:
            nb_to_add = self.batch_size // 2

        if len(self.history_buffer) < self.max_size:
            data_points_copy = np.copy(data_points)
            np.random.shuffle(data_points_copy)
            self.history_buffer = np.append(self.history_buffer, data_points_copy[:nb_to_add], axis=0)
        elif len(self.history_buffer) == self.max_size:
            self.history_buffer[:nb_to_add] = data_points[:nb_to_add]
        else:
            assert False

        np.random.shuffle(self.history_buffer)

    def get(self, nb_to_get=None):
        """
        Get a random sample of data points from the history buffer.

        :param nb_to_get: Number of data points to get from the history buffer (batch_size / 2 by default).
        :return: A random sample of `nb_to_get` data points from the history buffer, or an empty np array if the data
                history buffer is empty.
        """
        if not nb_to_get:
            nb_to_get = self.batch_size // 2

        try:
            return self.history_buffer[:nb_to_get]
        except IndexError:
            return np.zeros(shape=0)
    
    def is_empty(self):
        return len(self.history_buffer) <= 0

class SimGANDataset():
    """
    Holds a simulated and real dataset, each composed of 1D signals
    """

    def __init__(self, real_size=512, sim_size=128**2, batch_size=128, buffer_size=12800):
        self.batch_size = batch_size
        
        ### Get the real and simulated datasets
        # Generate Datasets:
        # Dataset  | Peak Locations | Ratio of Normed Peak Amplitudes | Frequency Content (Hz) in [Peak 1],[Peak 2]
        # -------------------------------------------------------------------------------------------------------------
        # REAL     |     24,56      |          1:0.64				  |              [1,4,6],[5,8,10]   
        # SYNTHETIC|    23.5,55.5   |          1:0.78                 |                [4]  ,  [10]  
        # Can configure these, but they are a bit obtuse
        self.real_raw = self.gen_fake_dataset(real_size, [24,56], [0.5,0.5], [5,8], [0.5,0.5], [1,0.64], [0.1,0.2], [[1,4,6],[5,8,10]], [0.25,0.25], True)
        self.simulated_raw = self.gen_fake_dataset(sim_size, [23.5,55.5], [0.5,0.5], [5,10], [0.5,0.5], [1,0.78], [0.05,0.17], [[4],[10]], [0.25,0.25], False)
        ### Get the labels
        self.labels_real = np.zeros(self.real_raw.shape[0])
        self.labels_simulated = np.ones(self.simulated_raw.shape[0])

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

        self.data_history_buffer = DataHistoryBuffer((1, self.real_raw.shape[-1]), buffer_size, batch_size)

        # a lot of times, we want to call .shape just to get num_channels and length of data so we cheat by doing this and doing None for num data
        _, real_channels, real_length = self.real.data.shape
        _, sim_channels, sim_length = self.simulated.data.shape
        assert(real_channels==sim_channels), "Something wrong with shape of data...mismatch number of channels"
        assert(real_length==sim_length), "Something wrong with shape of data...mismatch length of data"
        self.shape = (None, real_channels, real_length)
    
    def get_real_batch():
        return self.real_loader.__iter__().next()
    
    def get_simulated_batch():
        return self.simulated_loader.__iter__().next()
    
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

class TransformSimGANDataset(SimGANDataset):
    def __init__(self,
                 real_size=512,
                 sim_size=128**2,
                 batch_size=128,
                 buffer_size=12800):
        super().__init__(real_size=real_size,
                         sim_size=sim_size,
                         batch_size=batch_size,
                         buffer_size=buffer_size)
        #transform
        self.real_raw = self.transform(self.simulated_raw, real_size)
        
        #bookkeeping since we modified self.real
        self.real = dataset(self.real_raw, self.labels_real)
        self.real_loader = DataLoader(
            self.real,
            batch_size,
            shuffle=True,
            # **kwargs
        )
        self.data_history_buffer = DataHistoryBuffer((1, self.real_raw.shape[-1]), buffer_size, batch_size)
        _, real_channels, real_length = self.real.data.shape
        _, sim_channels, sim_length = self.simulated.data.shape
        assert(real_channels==sim_channels), "Something wrong with shape of data...mismatch number of channels"
        assert(real_length==sim_length), "Something wrong with shape of data...mismatch length of data"
        self.shape = (None, real_channels, real_length)
 
    def transform(self, dataset, real_size):
        dim_size = dataset[0].shape
        w = torch.rand([dim_size[-1], dim_size[-1]])
        b = torch.rand(dim_size[-1])
        i = torch.tensor(dataset).float()
        return F.linear(i, w, bias=b).numpy()[:real_size]


