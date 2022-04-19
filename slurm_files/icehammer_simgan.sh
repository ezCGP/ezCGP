#!/bin/bash
#SBATCH --job-name=ezCGP_simgan
#SBATCH -N 1                                         # number of machines
#SBATCH -c 8                                         # number of cores
#SBATCH --gres=gpu:1                                 # number of gpus
#SBATCH -C TeslaV100S-PCIE-32GB                       # QuadroRTX4000
#SBATCH -t 5-08:00                                   # job will run at most 8 hours D-HH:MM
#SBATCH --mem=128gb                                  # memory limit
#SBATCH --output=simgant.out # output file relative to ezcgp folder
echo "Started on `/bin/hostname`" # prints name of compute node job was started on

nvidia-smi

if [ -z "$1" ]
then
      setseed=""
else
      setseed="--seed $1"
fi

if [ -z "$2" ]
then
      setprevrun=""
else
      setprevrun="--previous_run $2"
fi

cd ~/ezCGP
module load anaconda3/2020.02
module load cuda/10.1
conda activate simgan-cgp
rm -rf ~/.nv
python main.py -p problem_simgan -v $setseed $setprevrun
