'''
It is hard to find a good way to get the memory footprint of a variable in python!
Here is my attempt to google around and steal code to do this.
'''
### packages
import gc

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

def getsizeof(input_obj):
    '''
    YO, this sucks

    https://towardsdatascience.com/the-strange-size-of-python-objects-in-memory-ce87bdfbb97f

    I think this returns bytes...note there are 1000000 bytes in MB
    '''
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    return memory_size

