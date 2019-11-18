#!/usr/bin/env bash
#mpiexec -n 1 ~/anaconda3/envs/ezCGP/bin/python mpi_universe.py
#OMP_NUM_THREADS=1 KMP_WARNING=OFF mpiexec.mpich -n 20 ~/anaconda3/envs/vip-hpc/bin/python mpi_universe.py
#OMP_NUM_THREADS=6 KMP_WARNING=OFF mpiexec -n 20 ~/anaconda3/envs/ezCGP/bin/python mpi_universe.py
#OMP_NUM_THREADS=8 KMP_WARNING=OFF mpiexec -n 20 ~/anaconda3/envs/ezCGP/bin/python mpi_universe.py
#python main.py
OMP_NUM_THREADS=1 KMP_WARNING=OFF OMP_PROC_BIND=FALSE mpiexec -n 4 python mpi_universe.py
