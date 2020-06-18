'''
If we're going to use mpi4py and logging, we need a way for the different nodes to use logging in sync.

What has been recommended is to create our own logging.FileHandler class. There is still a lot for me to learn
about this stuff but there have been these 2 solid examples to work from...the second is actually an iteration
off of the first.

1.) https://gist.github.com/JohnCEarls/8172807
2.) https://gist.github.com/chengdi123000/42ec8ed2cbef09ee050766c2f25498cb

how to use:
    comm = MPI.COMM_WORLD                                                           
    logger = logging.getLogger("rank[%i]"%comm.rank)                                
    logger.setLevel(logging.DEBUG)                                                  
                                                                                    
    mh = MPIFileHandler("logfile.log")                                           
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    mh.setFormatter(formatter)                                                                                                                             
    logger.addHandler(mh)     

    logger.debug('debug message') 
'''

### packages
from mpi4py import MPI
import logging
from os.path import abspath



class MPIFileHandler(logging.FileHandler):
	def __init__(self,
				 filename,
				 mode=MPI.MODE_WRONLY|MPI.MODE_CREATE|MPI.MODE_APPEND,
				 encoding='utf-8', #or None?
				 delay=False,
				 comm=MPI.COMM_WORLD):
		self.baseFilename = filename #inherited attribute...don't change the attribute name
		self.mode = mode
		self.encoding = encoding
		self.comm = comm

		if delay:
			'''
			honestly don't really get this...
			like we want a delay between the call and when we write to file? why tho
			"We don't open the stream, but we still need to call the Handler constructor to set level, formatter, lock etc."
			'''
			logging.Handler.__init__(self)
			self.stream = None
		else:
			# interesting that it's a logging.FileHandler being passed to logging.StreamHandler
			logging.StreamHandler.__init__(self, self._open())


	def _open(self):
		'''
		overwriting the _open of FileHandler which sets up a stream that we pass as input to StreamHandler
		'''
		stream = MPI.File.Open(self.comm, self.baseFilename, self.mode)
		stream.Set_atomicity(True)
		return stream


	def emit(self, record):
		'''
		overwriting the emit method of FileHandler and StreamHandler
		(technically StreamHandler.emit overwrit FileHandler.emit so we are only overwriting StreamHandler.emit)
		here is what the other guy put for emit:
		"If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        
        Modification:
            stream is MPI.File, so it must use `Write_shared` method rather
            than `write` method. And `Write_shared` method only accept 
            bytestring, so `encode` is used. `Write_shared` should be invoked
            only once in each all of this emit function to keep atomicity."
		'''
		try:
			msg = self.format(record)
			#print("yo", record)
			stream = self.stream
			stream.Write_shared((msg+self.terminator).encode(self.encoding))
			#self.flush #copied code had this commented out
		except Exception:
			#print("damn")
			self.handleError(record)


	def close(self):
		if self.stream:
			self.stream.Sync()
			self.stream.Close()
			self.stream = None