# ezCGP
A simple low level framework to Cartesian Genetic Programming (CGP) in Python(3).

### Design Doc
https://docs.google.com/document/d/1X8jGDXHAKkMBgOCYCtgT5v-wSqSLjVhxtZnGqr2hwz4/edit?usp=sharing


### Running locally
#### To run normal ezCGP
* For test run, `python tester.py`
* `python main.py`

#### To run parallel, with MPI
* Install mpi4py by running
`pip install mpi4py`
* To run mpi, locate where `mpiexec` is, and run this command
`mpiexec -n 4 python mpi_universe.py`   
The number `4` in the command above indicates how many CPUs you want to use, and `mpi_universe.py` is our parallelized version of `main.py`
