'''
play with different ways we could load in a NN architecutre and maybe even pipeline operators

going to import problem_cifar_no_transfer for convenience to focus only on the seeding stuff
'''

### packages
import numpy as np
from copy import deepcopy
import tensorflow as tf
import pdb

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from problems import problem_cifar_no_transfer
from codes.block_definitions.utilities import operators_TensorFlow_Keras as opTF
from codes.block_definitions.utilities import argument_types


def CNN_to_lisp(model):
    '''
    assumes tf.keras.applications Model
    mimics block_definition.get_lisp() especially for the args.
    and assumes model has no funky connections and is basically a linear connection
    '''
    lisp = "-1n"
    for i, layer in enumerate(model.layers[1:]):
        previous_lisp = deepcopy(lisp)

        # Conv2D
        if isinstance(layer, tf.keras.layers.Conv2D):
            operator = opTF.conv2D_layer
            arg_types = opTF.operator_dict[operator]["args"]
            args = []
            args.append(str(layer.filters))
            args.append(str(layer.kernel_size[0]))
            args.append(str(layer.activation))

        # MaxPool2D
        elif isinstance(layer, tf.keras.layers.MaxPool2D):
            operator = opTF.max_pool_layer
            arg_types = opTF.operator_dict[operator]["args"]
            args = []
            args.append(str(layer.pool_size[0]))
            args.append(str(layer.pool_size[1]))
            args.append(str(layer.strides[0]))

        else:
            print("not sure what this layer is...")
            pdb.set_trace()


        lisp = [operator.__name__, previous_lisp, *args]
    
    return lisp



class Problem(problem_cifar_no_transfer.Problem):
    def __init__(self):
        super().__init__()

        # now overwrite attributes
        self.population = 4
        
        pretrained_model = tf.keras.applications.VGG16(include_top=False,
                                                       weights='imagenet',
                                                       input_tensor=None,
                                                       pooling=None)
        block_lisp = CNN_to_lisp(pretrained_model)
        self.genome_seeds.append([None,None,block_lisp])
        '''
        pretrained_model = tf.keras.applications.ResNet101V2(include_top=False,
                                                             weights='imagenet',
                                                             input_tensor=None,
                                                             pooling=None
                                                            )
        self.genome_seeds.append(CNN_to_lisp(pretrained_model))

        pretrained_model = tf.keras.applications.ResNet152V2(include_top=False,
                                                             weights='imagenet',
                                                             input_tensor=None,
                                                             pooling=None
                                                            )
        self.genome_seeds.append(CNN_to_lisp(pretrained_model))

        pretrained_model = tf.keras.applications.ResNet50V2(include_top=False,
                                                            weights='imagenet',
                                                            input_tensor=None,
                                                            pooling=None
                                                            )
        self.genome_seeds.append(CNN_to_lisp(pretrained_model))'''