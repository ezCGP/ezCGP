#!/bin/bash
#SBATCH --job-name=ezCGP_mpi_test                    # name of job
#SBATCH -N 6                                         # number of machines...has to be same number for mpiexec -n
#SBATCH -c 4                                         # number of cores
#SBATCH -C "QuadroP4000|QuadroRTX4000"               # request specific gpus
#SBATCH --gres=gpu:1                                 # number of gpus
#SBATCH -t 0-08:00                                   # job will run at most 8 hours D-HH:MM
#SBATCH --output=mpi_test.out                        # output file relative to ezcgp folder
#SBATCH --mail-type=ALL                              # Will send a status email based on any combination of a,b,e
#SBATCH --mail-user=rtalebi3@gatech.edu              # Where to send email updates to
echo "Started on `/bin/hostname`"  # prints name of compute node job was started on


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
mpiexec -n 6 python main.py -p problem_multiGaussian_mpi -t -v $setseed $setprevrun
