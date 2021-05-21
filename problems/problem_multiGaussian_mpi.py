'''
root/problems/problem_multiGaussian.py
'''

### packages
import os
import numpy as np
import logging

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems import problem_multiGaussian
from post_process import save_things
from post_process import plot_things



class Problem(problem_multiGaussian.Problem):
    def __init__(self):
        super().__init__()
        self.mpi = True

    def plot_custom_stats2(self, folders):
        import glob
        import matplotlib.pyplot as plt

        if (type(folders) is str) and (os.path.isdir(folders)):
            '''# then assume we are looking for folders within this single folder
            poss_folders = os.listdir(folders)
            folders = []
            for poss in poss_folders:
                if os.path.isdir(poss):
                    folders.append(poss)'''
            # now that we are using glob below, we are all good...just make this into a list
            folders = [folders]
        elif type(folders) is list:
            # then continue as is
            pass
        else:
            print("we don't know how to handle type %s yet" % (type(folders)))

        # now try to find 'custom_stats.npz' in the folders
        stats = {}
        for folder in folders:
            npzs = glob.glob(os.path.join(folder,"*","custom_stats.npz"), recursive=True)
            for npz in npzs:
                data = np.load(npz)
                #genome_size = data['genome_size'][0]
                if folder not in stats:
                    stats[folder] = {'ids': [],
                                          'scores': [],
                                          'active_count': []}
                for key in ['ids','scores','active_count']:
                    stats[folder][key].append(data[key])

        # now go plot
        #plt.figure(figsize=(15,10))
        matplotlib_colors = ['b','g','r','c','m','y']
        fig, axes = plt.subplots(2, 1, figsize=(16,8))
        for ith_size, size in enumerate(stats.keys()):
            for row, key in enumerate(['scores','active_count']):
                datas = stats[size][key]
                for ith_data, data in enumerate(datas):
                    if key is 'scores':
                        data = data[:,0]
                    kwargs = {'color': matplotlib_colors[ith_size],
                              'linestyle': "-",
                              'alpha': 0.5}
                    if (row == 1) and (ith_data == 0):
                        kwargs['label'] = os.path.basename(size)
                    axes[row].plot(data, **kwargs)
        axes[row].legend()
        plt.show()
        import pdb; pdb.set_trace()
