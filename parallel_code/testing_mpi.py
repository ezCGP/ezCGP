from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


data = []
if rank == 0:
    for i in range(4):
        data.append([x for x in range(250)])
else:
    data = None

data = comm.scatter(data, root=0)

# parallelize

for i in range(len(data)):
    data[i] = data[i] * rank

data = comm.gather(data, )
print("CPU: " + str(rank), data)

