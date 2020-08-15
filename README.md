# ezCGP
A simple low level framework to Cartesian Genetic Programming (CGP) in Python(3).

### Design Doc
https://docs.google.com/document/d/1X8jGDXHAKkMBgOCYCtgT5v-wSqSLjVhxtZnGqr2hwz4/edit?usp=sharing

### Running locally
#### To run normal ezCGP
* For test run, `python tester.py`
* `python main.py`

#### To run parallel, with MPI
Get the MPI implementation in C on your machine first.  

Linux: Run `sudo apt install mpich`  
Windows: https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi?redirectedfrom=MSDN   
Mac: Run `brew install mpich`  

Then, install the python wrapper for MPI
* Install mpi4py by running
`pip install mpi4py`
* To run mpi, locate where `mpiexec` is, and run this command
`sh run.sh`   
The number `4` in the command above indicates how many CPUs you want to use, and `mpi_universe.py` is our parallelized version of `main.py`

### Environment Setup
For environmental consistency and to remove package conflicts with various OS's, please have Anaconda installed (e.g. version 4.7).

If you do not have Anaconda, please follow this [Digital Ocean guide](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart) to **step 8**.

**NOTE:** The requirements.txt file should only be used with the `conda install` command, **not pip install**.

> conda create -n ezCGP python=3.6

> conda activate ezCGP

> conda config --env --add channels menpo

> conda config --env --add channels conda-forge

> conda install --file requirements.txt

To activate your ezCGP Anaconda environment, simply run:

> conda activate ezCGP

You may also have to install another dependency depending on your machine. However, please **try running WITHOUT these packages first**:

Linux:

> sudo apt install mpich
> sudo apt install libgl1-mesa-glx

MacOS:

> brew install mpich

Windows: 

> https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi?redirectedfrom=MSDN   

### Running locally
#### To run sequential ezCGP 
* For test run: `python tester.py`
* For a full run: `python main.py`

Currently, sequential ezCGP has not been updated to include mating and many of the latest changes. We strongly recommend running parallel ezCGP.

#### To run parallel ezCGP
To run MPI:

1. Locate where `mpiexec` is 
2. Adjust `run.sh` with the mpiexec location
3. Run this command: `sh run.sh`

`mpi_universe.py` is our parallelized and latest version of `main.py`.

**NOTE:** The number in the `run.sh` file after `mpiexec -n` indicates how many CPU processes you want to use. 
