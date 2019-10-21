"""This File allows users to specify whether or not logs should be written to stdout or a log file
    If stdout is true logging will print to stdout
    if toFile is true logging will print to file
    this shouldn't cause pickling errors but the log files will be messy with mpi.
    Cleaner logging that supports mpi can be added later.
    To use the logger just import LogSetup.py main or tester before other ezCGP imports in the file you are about to run.
        For example, you would use 'import utils.LogSetup' in main.py and tester.py because you run these files from the commandline
    In all other files you just 'import logging' 
    To use the logger just type "logger.info(<message>)" or "logger.debug(<message>) etc.
    More information about python logging can be found at https://docs.python.org/3/library/logging.html
"""
import logging
import tensorflow as tf
tf.get_logger().setLevel('INFO')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #disables tf log outs


stdout = True
toFile = True #set to true if you want it to be redirected to File

#logging does not work on parrot os distr unless i do this
logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.NOTSET)

#logging params
maxSize = 1.5 * 10 **7
rolloverSize = 1


logPath = "Logs"
fileName = "ezCGPOut"
handlers = []
if stdout:
    handlers.append(logging.StreamHandler())
else:
    logging.propogate = False
    # handlers.append(logging.StreamHandler(stream = None))

if toFile:
    path = "{0}/{1}.log".format(logPath, fileName)
    file_ = open(path, "w") #clears output file
    file_.close()
    handlers.append(logging.FileHandler(path))

if stdout or toFile:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers = handlers)
