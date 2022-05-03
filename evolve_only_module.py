'''
Working on a project that wants to have ezCGP 'plugged into' another piece of code
that will handle the scoring, selection, and termination of evolution; let's call
that code the 'regulator'.
My code should just be plugged in to handle the evolution of the population only.
ezCGP needs to be imported into the regulator, a class loaded and instantiated
to handle the evolution and a run method to trigger a single generation of 
evolution.

Strategy:
* make a class to mimic what main.main() does
* make sure there is an __init__ and run method
'''
### packages
import os
import time
import numpy as np
import random
import tempfile
import logging
import gc

### absolute imports


class ezCGP_Module():
	def __init__(self, config_file, problem_file, seed):
		'''
		* high level, going to mimic what main.main() does
		* further will try to digest some config file to help
			dynamically set Problem() variables/setting
		'''
