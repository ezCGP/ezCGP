'''
How to get .npy data files for SimGAN ECG run.

https://github.com/ezCGP/ezCGP/issues/267#issuecomment-1121361872

Originally, we produced the .npy files via two .ipynb files outside of ezCGP.
This file pulls in the relevant code from each of those files so that it is 
more accessible to external users to test our SimGAN code.

In doing so we realized that neurokit2 is required here but isn't part of our
conda environment...so at least here are instructions for installing it via
https://neurokit2.readthedocs.io/en/latest/introduction.html#installation
    `conda install -c conda-forge neurokit2`

...same with wfdb
https://pypi.org/project/wfdb/
    `pip install wfdb`
'''

### packages
import os
import copy as cp
import time
import glob
import random
import numpy as np
import pandas as pd
from collections import Counter
import neurokit2 as nk
import wfdb


def gen_sim_dataset(save_dir="./"):
    '''
    COPY/PASTE FROM:
    misc/simgan_supplements/gen_ecg_sim.ipynb

    Generate a dataset of simulated ecg signals from both simulators
    Note this is actually a fairly small dataset, as the simulator output 
    is theoretically infinite is we don't account for similarity
    Currently generates 5000 samples, used for the GECCO 2022 publication
    '''
    sim_dataset = []
    for heart_rate in range(50, 100): # Vary heart rate
        '''
        A normal resting heart rate for adults ranges from 60 to 100 beats per minute. 
        Generally, a lower heart rate at rest implies more efficient heart function and 
        better cardiovascular fitness. For example, a well-trained athlete might have a 
        normal resting heart rate closer to 40 beats per minute.
        '''
        for noise in range(10): # vary noise percentage
            for seed in range(5): # vary seed
                ecg_com = nk.ecg_simulate(duration=10, noise=(noise*0.01), heart_rate=heart_rate, method="ecgsyn", sampling_rate=360, random_state=seed)
                sim_dataset.append(ecg_com)
                
                ecg_simple = nk.ecg_simulate(duration=10, noise=(noise*0.01), heart_rate=heart_rate, method="simple", sampling_rate=360, random_state=seed)
                sim_dataset.append(ecg_simple)
                    
    sim_dataset = np.array(sim_dataset)
    #print("Length: ", len(sim_dataset))
    np.save(os.path.join(save_dir,"sim_dataset"), sim_dataset, allow_pickle=True)
    return 


def read_file(file, participant, selected_labels):
    """
    Reads the original data files from the database and extracts the raw signal
    and the beat annotation labels for the RPeaks (vertical line in ECG readings).
    The signals are a half-hour long recording using a sampling rate of 360 per particpant (n=48)
    The specific explaination for the annotation symbols are found here
    https://archive.physionet.org/physiobank/annotations.shtml
    "+" is the start of the reading
    """
    # Get signal
    data = pd.DataFrame({"ECG": wfdb.rdsamp(file[:-4])[0][:, 0]})
    # getting annotations
    anno = wfdb.rdann(file[:-4], 'atr')
    symbols = [x for x in anno.symbol if x in selected_labels]
    anno = np.unique(anno.sample[np.isin(anno.symbol, selected_labels)])
    anno = pd.DataFrame({"Rpeaks": anno, "Symbol": symbols})

    return data, anno


def chunks(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def create_intervals(data_files, sampling_rate, selected_labels, save_dir="./"):
    """
    This involves splitting the hour long recording into 10 second segments 
    and creating a singular label for the classification of that segment
    For our purposes, we are selecting the majority non-normal annotation as the label
    [N, N, N, A, A, A, A, A, F] would select 'A' as the label for the segment
    
    If the given sequence is not the correct length, pad with zeros
    """
    dfs_ecg = []
    dfs_labels = []
    for participant, file in enumerate(data_files):
        print("Participant: " + str(participant + 1) + "/" + str(len(data_files)))

        data, anno = read_file(file, participant, selected_labels)
        segments = list(chunks(data["ECG"], sampling_rate*10))
        labels = []
        for i, segment in enumerate(segments):
            segments[i] = segment.to_numpy()
            if len(segment) != sampling_rate*10:
                segments[i] = np.pad(segment, (0, sampling_rate*10 - len(segment)), 'constant')
            start = i*sampling_rate*10 
            end = start+sampling_rate*10 
            idx = np.where((anno["Rpeaks"]<end) & (anno["Rpeaks"] >= start))
            beat_annotations = anno.loc[idx]['Symbol']
            beat_annotations_copy = cp.deepcopy(beat_annotations)
            while "N" in beat_annotations_copy:
                beat_annotations_copy.remove("N")
            if len(beat_annotations_copy) == 0:
                label = "N"
            else:
                label = Counter(beat_annotations).most_common(1)[0][0]
            labels.append(label)
        dfs_ecg.extend(segments)
        dfs_labels.extend(labels)

        # Store additional recording if available
        xfile = os.path.join(save_dir, "physionet.org/files/mitdb/1.0.0/x_mitdb", "x_%s" % os.path.basename(file))
        if os.path.exists(xfile):
            print("  - Additional recording detected.")
            data, anno = read_file(xfile, participant, selected_labels)
            # Store with the rest
            segments = list(chunks(data["ECG"], sampling_rate*10))
            labels = []
            for i, segment in enumerate(segments):
                segments[i] = segment.to_numpy()
                if len(segment) != sampling_rate*10:
                    segments[i] = np.pad(segment, (0, sampling_rate*10 - len(segment)), 'constant')
                start = i*sampling_rate*10 
                end = start+sampling_rate*10 
                idx = np.where((anno["Rpeaks"]<end) & (anno["Rpeaks"] >= start))
                beat_annotations = anno.loc[idx]['Symbol']
                beat_annotations_copy = cp.deepcopy(beat_annotations)
                while "N" in beat_annotations_copy:
                    beat_annotations_copy.remove("N")
                if len(beat_annotations_copy) == 0:
                    label = "N"
                else:
                    label = Counter(beat_annotations).most_common(1)[0][0]
                labels.append(label)
            dfs_ecg.extend(segments)
            dfs_labels.extend(labels)

    # Save
    dfs_ecg = np.stack(dfs_ecg, axis=0)
    dfs_labels = np.array(dfs_labels)
    #np.save(os.path.join(save_dir, "ECGs"), dfs_ecg, allow_pickle=True)
    #np.save(os.path.join(save_dir, "labels"), dfs_labels, allow_pickle=True)
    return dfs_ecg, dfs_labels


def seperate_refiner_binary_classes(dfs_ecg, dfs_labels, save_dir="./"):
    """
    COPY/PASTE FROM:
    misc/simgan_supplements/read_mit_bih.ipynb

    Seperates the signals based on normal vs abnormal heartbeat
    Expected use case it to feed the examples to a refiner to train
    on general normal or abnormal heartbeats
    """
    norm_idx = np.where(dfs_labels == "N")
    ab_idx = np.where(dfs_labels != "N")
    dfs_ecg_normal = dfs_ecg[norm_idx]
    dfs_ecg_abnormal = dfs_ecg[ab_idx]
    np.save(os.path.join(save_dir, "normal"), dfs_ecg_normal, allow_pickle=True)
    np.save(os.path.join(save_dir, "abnormal"), dfs_ecg_abnormal, allow_pickle=True)
    return


def get_real_dataset(save_dir="./"):
    '''
    NOT copied over for ipynb but made to combine all the other methods into 1 convenient call
    '''
    download_real_cmd = "wget -P %s -r -N -c -np https://physionet.org/files/mitdb/1.0.0/ --no-check-certificate" % data_dir
    os.system(download_real_cmd)
    time.sleep(1)

    data_files = glob.glob(os.path.join(save_dir, "physionet.org/files/mitdb/1.0.0/*dat"))
    selected_labels = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
    dfs_ecg, dfs_labels = create_intervals(data_files, 360, selected_labels, save_dir)

    # Due to the small number of samples for some classes, we will remove them before partitioning
    # them into train and test sets
    #print("Before: ", len(dfs_ecg))
    #print(pd.DataFrame({"label": dfs_labels}).value_counts())
    for label in ["a", "Q"]:
        while label in dfs_labels:
            idx = np.where(dfs_labels == label)[0][0]
            dfs_ecg = np.delete(dfs_ecg, idx, axis=0)
            dfs_labels = np.delete(dfs_labels, idx)
    #print("After: ", len(dfs_ecg))

    seperate_refiner_binary_classes(dfs_ecg, dfs_labels, save_dir)



if __name__ == "__main__":
    np.random.seed(seed=0)
    random.seed(0)

    ezcgp_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    data_dir = os.path.join(ezcgp_root_dir, 'data', 'datasets', 'ecg')
    os.makedirs(data_dir, exist_ok=True)
    
    gen_sim_dataset(save_dir=data_dir)
    get_real_dataset(save_dir=data_dir)
