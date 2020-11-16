# ezCGP
A low level framework for Cartesian Genetic Programming in Python3.

Be sure to checkout the [Wiki](https://github.com/ezCGP/ezCGP/wiki) for more documentation and helpful guids.

# Quickstart
Create environment and download dependencies
```
$ conda create -n ezcgp-py --file conda_environment.yaml
$ conda activate ezcgp-py
$ pip install Augmentor
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

test push from local

