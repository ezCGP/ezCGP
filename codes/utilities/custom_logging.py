'''
I hope this isn't overkill but I'm planning on creating a wrapper class around logging.

Basically the only point of this is to weave in MPI so that I can easily grab the noderank
and call the proper logger from logging.getLogger(name="")

ezLogging.info
'''

### packages
import logging
from mpi4py import MPI

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root



class ezLogging():
    '''
    there is no __init__ method so anything here that refers to itmust call the class name as well
    '''
    def get_logger():
        '''
        this is the most important part of the whole class really

        first tried to see if I can do ```'MPI' in globals()``` so that I don't always have to 
        import MPI but I think we have to accept it and force MPI all the time
        
        if 'MPI' in globals():
            print("True")
            log_name = "NodeRank-%i" % MPI.COMM_WORLD.Get_Rank()
        else:
            print("False")
            #log_name = None # for RootLogger
            log_name = "NodeRank-0"'''
        log_name = "NodeRank%i" % MPI.COMM_WORLD.Get_rank()
        return logging.getLogger(log_name)


    def debug(msg):
        my_log = ezLogging.get_logger()
        my_log.debug(msg)


    def info(msg):
        my_log = ezLogging.get_logger()
        my_log.info(msg)


    def warning(msg):
        my_log = ezLogging.get_logger()
        my_log.warning(msg)


    def error(msg):
        my_log = ezLogging.get_logger()
        my_log.error(msg)


    def critical(msg):
        my_log = ezLogging.get_logger()
        my_log.critical(msg)


    def logging_setup(loglevel):
        my_log = ezLogging.get_logger()
        # DEBUG -> INFO -> WARNING -> ERROR -> CRITICAL
        my_log.setLevel(loglevel)
        # set format of how every single log entry will start with
        #format_str = "[%(asctime)s.%(msecs)d][%(threadName)s-%(thread)d][%(filename)s-%(funcName)s] %(levelname)s: %(message)s"
        # removed [%(filename)s-%(funcName)s] because now it only gives "custom_logging.py"
        format_str = "[%(asctime)s.%(msecs)d][%(name)s] %(levelname)s: %(message)s"
        log_formatter = logging.Formatter(fmt=format_str, datefmt="%H:%M:%S")
        return log_formatter


    def logging_2stdout(log_formatter):
        log_handler_2stdout = logging.StreamHandler(sys.stdout)
        log_handler_2stdout.setFormatter(log_formatter)
        my_log = ezLogging.get_logger()
        my_log.addHandler(log_handler_2stdout)
        return log_handler_2stdout


    def logging_2file(log_formatter, filename):
        log_handler_2file = logging.FileHandler(filename, 'w')
        log_handler_2file.setFormatter(log_formatter)
        my_log = ezLogging.get_logger()
        my_log.addHandler(log_handler_2file)
        return log_handler_2file


    def logging_2file_mpi(log_formatter, filename):
        from codes.utilities import mpi_logging_helper
        log_handler_2file = mpi_logging_helper.MPIFileHandler(filename)
        log_handler_2file.setFormatter(log_formatter)
        my_log = ezLogging.get_logger()
        my_log.addHandler(log_handler_2file)
        return log_handler_2file


    def logging_remove_handler(handler):
        my_log = ezLogging.get_logger()
        my_log.removeHandler(handler)