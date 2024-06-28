from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import random
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib
import os
from copy import deepcopy
from PIL import Image
from typing import List


### sys relative to root dir
import sys
from os.path import dirname, realpath
#sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from codes.factory import FactoryDefinition, Factory_SimGAN
from problems.problem_simgan import Problem

from tqdm import tqdm


class MyDataset(Dataset):
    '''
    A wrapper class for the torch.Dataset for pytorch training things
    '''
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)

def build_individual(individual_pkl):
    '''
    a lot of this is just going to mimic factory.Factory_SimGAN.build_individual_from_seed
    but use the trained weights instead!
    '''
    problem = Problem()
    indiv_def = problem.indiv_def
    factory = Factory_SimGAN()


    # try to grab id from name
    try:
        indiv_id = os.path.basename(individual_pkl).split("_")[-1].split(".")[0]
    except Exception as err:
        print("Couldn't get id for some reason:\n%s" % err)
        indiv_id = "poopie_face"

    maximize_objectives = [False, False, False, True]

    individual = factory.build_individual_from_seed(indiv_def=indiv_def, block_seeds=individual_pkl, maximize_objectives_list=maximize_objectives, indiv_id=indiv_id)

    refiner, discriminator = deepcopy(individual.output)
    del individual

    return refiner, discriminator

def generate_ecg_img(syn_batch, ref_batch, real_batch, png_path, ref_preds=None, real_preds=None):
    '''
    Generates matplotlib plots of 3 syn, ref, and real examples from the step.
    '''

    fig = plt.figure(figsize=(20, 15))

    ax1 = fig.add_subplot(3, 1, 1)
  
    print(syn_batch[0].numpy())
    ax1.plot(syn_batch[0].numpy())
    ax1.set_ylim([-0.8, 1.3])
    ax1.set_title("Random Simulated Signal", fontsize=25)
    ax2 = fig.add_subplot(3, 1, 2)

    ax2.plot(ref_batch[0].numpy())
    ax2.set_ylim([-0.8, 1.3])
    ax2.set_title("Refined Signal", fontsize=25)
    
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(real_batch[0].numpy(), color='orange')
    ax3.set_ylim([-0.8, 1.3])
    ax3.set_title("Random Real Signal", fontsize=25)
    plt.xlabel('Samples', fontsize=21)
    plt.ylabel('Amplitude (mV)', fontsize=21)

    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18) 

    plt.savefig(png_path, format='eps')

    plt.close()

def train_model(model, trainx, trainy):
    '''
    Trains sklearn models
    '''
    model.fit(trainx, trainy)

def test_model(model, testx, testy, lab, file_path):
    '''
    Tests all models using classification report and f1 score
    Assumes that it is binary classification between normal and abnormal heartbeats
    '''
    y_pred = model.predict(testx)
    print(classification_report(testy, y_pred))
    print(f1_score(testy, y_pred))
    print('----------------------------------')
    cf_matrix = confusion_matrix(testy, y_pred, normalize='all')
    plt.title(f"Confusion Matrix {lab}") # title with fontsize 20
    plt.xlabel('Predicted') # x-axis label with fontsize 15
    plt.ylabel('True') # y-axis label with fontsize 15
    sns.heatmap(cf_matrix, linewidths=1, annot=True, fmt='g', yticklabels=['Normal', 'Abnormal'], xticklabels=['Normal', 'Abnormal'])
    plt.savefig(file_path, format='eps')
    plt.close()


'''
The next few classes are a pytorch implementation of a 1D SqueezeNet originally
taken from https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py 
that was converted to a 1d network
'''
class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv1d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv1d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv1d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )


class SqueezeNet(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(1, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool1d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool1d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(512, 64, 256, 256),
        )
  
        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv1d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if m is final_conv:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        #print(x)
        #print(x.shape)
        return x

def train_test_squeezenet(x_train, y_train, dataloader_test, testy, lab, file_path):
    dataset = MyDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=256)
    # make model
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    model = SqueezeNet()
    model.to(device)

    # train and test
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    n_epoch = 20
    train_losses = []
    for epoch in range(n_epoch):
        train_loss = 0.0
        model.train()
        for data, target in dataloader:
            data = data.to(device).float()
            target = target.to(device).squeeze()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            #backward-pass
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Update the Training loss
            train_loss += loss.item() 
        #print(train_loss)

    # TODO: Plot train losses and train until convergence/early stopping
    # Originally just trained for a set hard coded value, if we want to continue this
    # for a journal publication we should train till convergence


    '''
    Note that sometimes the model gets caught in some weird optima 
    where it returns all zeros (likely due to class imbalances)
    Retrain the model if this occurs, and check the train losses to make sure it
    is learning during training
    '''
    model.eval()
    prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
    all_pred_prob = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(prog_iter_test):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = F.softmax(model(input_x), dim=1)
            all_pred_prob.append(pred.cpu().data.numpy())

    all_pred_prob = np.concatenate(all_pred_prob)
    all_pred = np.argmax(all_pred_prob, axis=1)
    labels = testy.flatten()
    print(classification_report(labels, all_pred))
    print(f1_score(labels, all_pred))
    print('----------------------------------')
    cf_matrix = confusion_matrix(labels, all_pred, normalize='all')
    plt.title(f"Confusion Matrix {lab}") # title with fontsize 20
    plt.xlabel('Predicted') # x-axis label with fontsize 15
    plt.ylabel('True') # y-axis label with fontsize 15
    sns.heatmap(cf_matrix, linewidths=1, annot=True, fmt='g', yticklabels=['Normal', 'Abnormal'], xticklabels=['Normal', 'Abnormal'])
    plt.savefig(file_path, format='eps')
    plt.close()

def main():

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    np.random.seed(seed=0)
    random.seed(0)

    # Pickle files from ezcgp runs
    refinerevo, _ = build_individual('gen_0009_indiv_2d311ddb156f0.pkl') # Evolved individual
    refinerseed, _ = build_individual('gen_0000_indiv_seededIndiv0-1.pkl') # Seeded individual

    # Note that to load the individuals you will also need the folder produced by ezCGP for each individual
    
    print("Finished Build Individuals")

    # ezCGP only uses the first 32 samples of real data for simgan training, due to FID memory issues
    # The first 32 real samples are therefore ignored for classification training
    # Once memory is fixed, this number may change
    abnormal = np.load('./data/datasets/abnormal.npy', allow_pickle=True)[32:] 
    normal = np.load('./data/datasets/normal.npy', allow_pickle=True)
    sim_dataset = np.load('./data/datasets/sim_dataset.npy', allow_pickle=True)
    print('read data')

    sim_loader = DataLoader(sim_dataset, batch_size=1, shuffle=False)
    seed_ref = []
    for i, data in enumerate(sim_loader):
        # get the inputs; data is a list of [inputs, labels]
        outputs = refinerseed(data.unsqueeze(1).float())
        seed_ref.append(outputs.detach().numpy().squeeze())
        
    seed_ref = np.array(seed_ref)
    print('got seeded refined')

    sim_loader = DataLoader(sim_dataset, batch_size=1, shuffle=False)
    
    evo_ref = []
    for i, data in enumerate(sim_loader):
        # get the inputs; data is a list of [inputs, labels]
        outputs = refinerevo(data.unsqueeze(1).float())
        evo_ref.append(outputs.detach().numpy().squeeze())
        
    evo_ref = np.array(evo_ref)
    print('got evolved refined')
    
    # make normal dataset, where a train/test split is made from real signals from MIT-BIH
    unsim_x = np.concatenate([abnormal, normal])
    unsim_y = np.concatenate([np.ones([abnormal.shape[0], 1]), np.zeros([normal.shape[0], 1])])

    unsim_x_train, unsim_x_test, unsim_y_train, unsim_y_test = train_test_split(unsim_x, unsim_y, test_size=0.25, random_state=42)
    unsim_x_train = np.expand_dims(unsim_x_train, axis=1)
    unsim_x_test = np.expand_dims(unsim_x_test, axis=1)
    print(unsim_x_train.shape)
    print(unsim_x_test.shape)

    # make simulated training dataset where it is a combination of real and 
    # simulated data, where the simulated waves are normal heartbeats (class=0)
    sim_x = np.concatenate([unsim_x_train, np.expand_dims(sim_dataset, axis=1)])
    sim_y = np.concatenate([unsim_y_train, np.zeros([sim_dataset.shape[0], 1])])
    sim_x, sim_y = shuffle(sim_x, sim_y)
    print(sim_x.shape)
    print(sim_y.shape)


    # make refined training dataset using seeded refiner where it is a combination of real and 
    # refined data, where the simulated waves are passed through the refiner
    # to be made into abnormal heartbeats (class=1)
    seedref_x = np.concatenate([unsim_x_train, np.expand_dims(seed_ref,axis=1)])
    seedref_y = np.concatenate([unsim_y_train, np.ones([sim_dataset.shape[0], 1])])
    seedref_x, seedref_y = shuffle(seedref_x, seedref_y)
    print(seedref_x.shape)
    print(seedref_y.shape)

    
    # make refined training dataset using evolved refiner where it is a combination of real and 
    # refined data, where the simulated waves are passed through the refiner
    # to be made into abnormal heartbeats (class=1)
    evoref_x = np.concatenate([unsim_x_train, np.expand_dims(evo_ref, axis=1)])
    evoref_y = np.concatenate([unsim_y_train, np.ones([sim_dataset.shape[0], 1])])
    evoref_x, evoref_y = shuffle(evoref_x, evoref_y)
    print(evoref_x.shape)
    print(evoref_y.shape)

    model_list = [DecisionTreeClassifier(random_state=42), GradientBoostingClassifier(random_state=42), MLPClassifier(random_state=42), AdaBoostClassifier(random_state=42)]


    for model in model_list:
        trained_model = deepcopy(model)
        model_name = type(trained_model).__name__
        print(model_name)
        train_model(trained_model, unsim_x_train, unsim_y_train)
        test_model(trained_model, unsim_x_test, unsim_y_test, 'No Simulated Data', '{}_real_confusion_matrix.eps'.format(model_name))

        train_model(trained_model, sim_x, sim_y)
        test_model(trained_model, unsim_x_test, unsim_y_test, 'Simulated But Not Refined Data', '{}_sim_confusion_matrix.eps'.format(model_name))

        train_model(trained_model, seedref_x, seedref_y)
        test_model(trained_model, unsim_x_test, unsim_y_test, 'Data Refined with Seed Model', '{}_seed_confusion_matrix.eps'.format(model_name))

        train_model(trained_model, evoref_x, evoref_y)
        test_model(trained_model, unsim_x_test, unsim_y_test, 'Data Refined with Evolved Model', '{}_evo_confusion_matrix.eps'.format(model_name))

    # Test SqueezeNet model
    dataset_test = MyDataset(unsim_x_test, unsim_y_test)
    dataloader_test = DataLoader(dataset_test, batch_size=256, drop_last=False)
    
    train_test_squeezenet(unsim_x_train, unsim_y_train, dataloader_test, unsim_y_test, 'Real Data', 'squeezenet_real_confusion_matrix.eps')

    train_test_squeezenet(sim_x, sim_y, dataloader_test, unsim_y_test, 'Real Data + Sim Data', 'squeezenet_sim_confusion_matrix.eps')
   
    train_test_squeezenet(seedref_x, seedref_y, dataloader_test, unsim_y_test, 'Real Data + Ref Data', 'squeezenet_seed_confusion_matrix.eps')

    train_test_squeezenet(evoref_x, evoref_y, dataloader_test, unsim_y_test, 'Real Data + Ref Data', 'squeezenet_evo_confusion_matrix.eps')
          


if __name__ == "__main__":
    main()
