#!/bin/bash
#SBATCH --job-name=train_stats                       # name of job
#SBATCH -N 1                                         # number of machines
#SBATCH -c 8                                         # number of cores
#SBATCH --gres=gpu:1                                 # number of gpus
#SBATCH -C TeslaV100-PCIE-32GB                       # QuadroRTX4000
#SBATCH -t 0-08:00                                   # job will run at most 8 hours D-HH:MM
#SBATCH --mem=128gb                                  # memory limit
#SBATCH --output=train_stats.out                     # output file relative to ezcgp folder
#SBATCH --mail-type=ALL                              # Will send a status email based on any combination of a,b,e
#SBATCH --mail-user=rtalebi3@gatech.edu              # Where to send email updates to
echo "Started on `/bin/hostname`" # prints name of compute node job was started on

nvidia-smi

problem="--problem $1"

cd ~/ezCGP/codes/utilities
module load anaconda3/2021.11
module load cuda/10.1
conda activate ezcgp-py
rm -rf ~/.nv
python train_network_stats.py $problem --module tensorflow
