# ezCGP
A low level framework for Cartesian Genetic Programming in Python3.

Be sure to checkout the [Wiki](https://github.com/ezCGP/ezCGP/wiki) for more documentation and helpful guids.

# Quickstart
Create environment and download dependencies
```
$ git clone https://github.com/ezCGP/ezCGP.git
$ cd ezcgp
$ conda env create -f python_envs/conda_environment.yml
$ conda activate ezcgp-py
$ conda deactivate
```

To run in a single process...
```
$ conda activate ezcgp-py
$ python main.py -p [problem_file].py
```

To run with MPI...
```
$ conda activate ezcgp-py
$ mpiexec -n [number_cores] python main.py -p [problem_file].py
```

Other arguments for main.py
* `-v` or `-d` to set the logging level to 'verbose' or 'debug'. The latter will not print to stdout.
* `-s [int]` to set the random seed
* `-t` to change the name of the output folder for this run to lead with the word 'testing'
* `-n [str]` to change the name of the output folder for this run to trail with a custom string tag

