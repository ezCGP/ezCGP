#!/usr/bin/env bash
# Please add your name right before your run script so that others wont modify it

#mpiexec -n 1 ~/anaconda3/envs/ezCGP/bin/python mpi_universe.py
#OMP_NUM_THREADS=8 KMP_WARNING=OFF mpiexec -n 20 ~/anaconda3/envs/ezCGP/bin/python mpi_universe.py
#python main.py

# Sam's MacOS script
OMP_NUM_THREADS=1 KMP_WARNING=OFF OMP_PROC_BIND=FALSE mpiexec -n 2 python3 mpi_universe.py

# Sam's PACE script
#OMP_NUM_THREADS=1 KMP_WARNING=OFF OMP_PROC_BIND=FALSE mpiexec -n 12 python3.6 mpi_universe.py

# Mai's script
#OMP_NUM_THREADS=1 KMP_WARNING=OFF mpiexec.mpich -n 2 ~/anaconda3/envs/ezCGP/bin/python mpi_universe.py
