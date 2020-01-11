from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


k = 1
while True:
    k *= 1.1

