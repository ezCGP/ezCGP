'''
simple. try to get modularity going on an 'advanced' toy problem.
'''

### global modules
import sys
import os
import numpy as np 
from copy import deepcopy
import time
from abc import ABC, abstractmethod
import Augmentor
import cv2
import tensorflow as tf
import pdb



####################################################################################################
### Modules

class Module_Abstract(ABC):
    def __init__(self, expected_inputs):
        '''
        allowed inputs: list of tuples
            * len of list is equal to required/expected number of inputs
            * each element is a tuple of all the types allowed for that input
        '''
        # just to make sure i didn't mess up on the syntax + data types
        if not isinstance(expected_inputs, list):
            print("ERROR BRO - wrong data type for expected_inputs")
            pdb.set_trace()
        for _input in expected_inputs:
            if not isinstance(ting, tuple):
                print("ERROR BRO - wrong data type for elements in expected_inputs")
                pdb.set_trace()

        self.num_inputs = len(expected_inputs)
        self.expected_inputs = expected_inputs
        self.transition = False


    @abstractmethod
    def is_transition(self, inputs):
        pass



### Modules - Input Nodes

class InputNode(Module_Abstract):
    def __init__(self):
        super().__init__(expected_inputs=[])


    def is_transition(self, inputs):
        self.transition = False



class Pipeline_InputNode(InputNode):
    def __init__(self):
        super().__init__()



class Images_InputNode(InputNode):
    def __init__(self):
        super().__init__()



### Modules - Main Nodes

class MainNode(Module_Abstract):
    def __init__(self, expected_inputs):
        super().__init__(expected_inputs)


    def is_transition(self, inputs):
        this_type = type(self).__name__

        exists = False
        for _input in inputs:
            if type(_input.module).__name__ == this_type:
                exists = True
                break

        if exists:
            self.transition = False
        else:
            self.transition = True



class Augment_MainNode(MainNode):
    def __init__(self):
        super().__init__(expected_inputs=[(Pipeline_InputNode, Augment_MainNode)])



class Preprocess_MainNode(MainNode):
    def __init__(self):
        super().__init__(expected_inputs=[(Augment_MainNodee, Preprocess_MainNode)])



class StartGraph_MainNode(MainNode):
    def __init__(self):
        super().__init__(expected_inputs=[(Pipeline_InputNode)])



class BuildGraph_MainNode(MainNode):
    def __init__(self):
        super().__init__(expected_inputs=[(StartGraph_MainNode, BuildGraph_MainNode)])



class EndGraph_MainNode(MainNode):
    def __init__(self):
        super().__init__(expected_inputs=[(StartGraph_MainNode),
                                          (BuildGraph_MainNode)])



class ImageGenerator_MainNode(MainNode):
    def __init__(self):
        super().__init__(expected_inputs=[(Images_InputNode),
                                          (Preprocess_MainNode)])



class TrainGraph_MainNode(MainNode):
    def __init__(self):
        super().__init__(expected_inputs=[(EndGraph_MainNode),
                                          (Pipeline_InputNode)])



class ScoreGraph_MainNode(MainNode):
    def __init__(self):
        super().__init__(expected_inputs=[(TrainGraph_MainNode)])



### Modules - Output Nodes

class OutputNode(Module_Abstract):
    def __init__(self, expected_inputs):
        super().__init__(expected_inputs)


    def is_transition(self, inputs):
        self.transition = True



class FinalOutput(OutputNode):
    def __init__(self):
        super().__init__(expected_inputs=[(ScoreGraph_MainNode)])



####################################################################################################
### Primtives

operator_dict = {}



### Primitives - Image Augmentation

def flip_left_right(pipeline, probability=.5):
    '''
    https://arxiv.org/pdf/1912.11370v2.pdf
    https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.flip_left_right
    "Flip (mirror) the image along its horizontal axis, i.e. from left to right."
    prob: float (0,1]
    '''
    pipeline.flip_left_right(probability=probability)
    return pipeline

operator_dict[flip_left_right] = {"module": Augment_MainNode,
                                  "inputs": [Augmentor.Pipeline],
                                  "output": Augmentor.Pipeline,
                                  "args": []
                                 }


def flip_top_bottom(pipeline, probability=.5):
    '''
    https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.flip_top_bottom
    "Flip (mirror) the image along its vertical axis, i.e. from top to bottom."
    prob: float (0,1]
    '''
    pipeline.flip_top_bottom(probability=probability)
    return pipeline

operator_dict[flip_top_bottom] = {"module": Augment_MainNode,
                                  "inputs": [Augmentor.Pipeline],
                                  "output": Augmentor.Pipeline,
                                  "args": []
                                 }



### Primitives - Image Preprocessing

class Blur(Augmentor.Operations.Operation):
    '''
    https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gad533230ebf2d42509547d514f7d3fbc3
    
    when normalize=True, same as cv2.blur()
    '''
    def __init__(self, kernel_size=5, normalize=True, probability=1):
        super().__init__(probability=probability)
        self.kernel = (kernel_size, kernel_size)
        self.normalize = normalize
    
    def perform_operation(self, images):
        def do(image):
            return cv2.blur(image, ksize=self.kernel, normalize=self.normalize)
        
        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        
        return augmented_images


def blur(pipeline, kernel_size=5, normalize=True):
    pipeline.add_operation(Blur(kernel_size, normalize))
    return pipeline


operator_dict[blur] = {"module": Preprocess_MainNode,
                       "inputs": [Augmentor.Pipeline],
                       "output": Augmentor.Pipeline,
                       "args": []
                      }


class BilateralFilter(Augmentor.Operations.Operation):
    '''
    https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
    
    * d: Diameter of each pixel neighborhood that is used during filtering.
        If it is non-positive, it is computed from sigmaSpace.
    * sigmaColor: Filter sigma in the color space. A larger value of the
        parameter means that farther colors within the pixel neighborhood
        (see sigmaSpace) will be mixed together, resulting in larger areas
        of semi-equal color.
    * sigmaSpace: Filter sigma in the coordinate space. A larger value of
        the parameter means that farther pixels will influence each other
        as long as their colors are close enough (see sigmaColor ). When d>0,
        it specifies the neighborhood size regardless of sigmaSpace. Otherwise, 
        d is proportional to sigmaSpace.
    
    Sigma values: For simplicity, you can set the 2 sigma values to be the same.
    If they are small (< 10), the filter will not have much effect, whereas if
    they are large (> 150), they will have a very strong effect, making the image
    look "cartoonish".

    Filter size: Large filters (d > 5) are very slow, so it is recommended to use
    d=5 for real-time applications, and perhaps d=9 for offline applications that
    need heavy noise filtering.
    '''
    def __init__(self, d, sigma_color, sigma_space, probability=1):
        super().__init__(probability=probability)
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
    
    def perform_operation(self, images):
        def do(image):
            return cv2.bilateralFilter(image, d=self.d, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)
        
        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        
        return augmented_images


def bilateral_filter(pipeline, d=2, sigma_color=20.0, sigma_space=20.0):
    pipeline.add_operation(BilateralFilter(d, sigma_color, sigma_space))
    return pipeline


operator_dict[bilateral_filter] = {"module": Preprocess_MainNode,
                                   "inputs": [Augmentor.Pipeline],
                                   "output": Augmentor.Pipeline,
                                   "args": []
                                  }



### Primitives - Init Graph

def init_graph(pipeline):
    input_layer = tf.keras.Input(shape=pipeline.image_shape,
                                 batch_size=pipeline.batch_size,
                                 dtype=None)
    return input_layer


operator_dict[init_graph] = {"module": StartGraph_MainNode,
                                   "inputs": [Augmentor.Pipeline],
                                   "output": Augmentor.Pipeline,
                                   "args": []
                                  }



### Primitives - Add to Graph

def conv2D_layer_relu(input_tensor, filters=32, kernel_size=5, activation=tf.nn.relu):
    kernel_size = (kernel_size, kernel_size)
    '''
    Convolutional Layer
    Computes 32 features using a 5x5 filter with ReLU activation.
    Padding is added to preserve width and height.

    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
    '''
    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding="same",
                                  activation=activation,
                                  data_format="channels_last"
                                 )(input_tensor)

operator_dict[conv2D_layer_relu] = {"module": BuildGraph_MainNode,
                               "inputs": [tf.keras.layers],
                               "output": tf.keras.layers,
                               "args": []
                              }


def conv2D_layer_tanh(input_tensor, filters=32, kernel_size=5, activation=tf.nn.tanh):
    kernel_size = (kernel_size, kernel_size)
    '''
    Convolutional Layer
    Computes 32 features using a 5x5 filter with ReLU activation.
    Padding is added to preserve width and height.

    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
    '''
    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding="same",
                                  activation=activation,
                                  data_format="channels_last"
                                 )(input_tensor)

operator_dict[conv2D_layer_tanh] = {"module": BuildGraph_MainNode,
                               "inputs": [tf.keras.layers],
                               "output": tf.keras.layers,
                               "args": []
                              }


def conv2D_layer_elu(input_tensor, filters=32, kernel_size=5, activation=tf.nn.elu):
    kernel_size = (kernel_size, kernel_size)
    '''
    Convolutional Layer
    Computes 32 features using a 5x5 filter with ReLU activation.
    Padding is added to preserve width and height.

    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
    '''
    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding="same",
                                  activation=activation,
                                  data_format="channels_last"
                                 )(input_tensor)

operator_dict[conv2D_layer_elu] = {"module": BuildGraph_MainNode,
                               "inputs": [tf.keras.layers],
                               "output": tf.keras.layers,
                               "args": []
                              }



### Primitives - End Graph

def close_graph(first_layer, last_layer):
    output_flatten = tf.keras.layers.Flatten()(last_layer)
    logits = tf.keras.layers.Dense(units=datapair.num_classes, activation=None, use_bias=True)(output_flatten)
    softmax = tf.keras.layers.Softmax(axis=1)(logits) # TODO verify axis...axis=1 was given by original code

    graph = tf.keras.Model(inputs=first_layer, outputs=softmax)

    graph.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss="categorical_crossentropy",
                 metrics=[tf.keras.metrics.Accuracy(),
                          tf.keras.metrics.Precision(),
                          tf.keras.metrics.Recall()],
                 loss_weights=None,
                 weighted_metrics=None,
                 run_eagerly=None)

    return graph


operator_dict[close_graph] = {"module": EndGraph_MainNode,
                                   "inputs": [Augmentor.Pipeline],
                                   "output": Augmentor.Pipeline,
                                   "args": []
                                  }



### Primitives - Build Generator

def build_generator(pipeline, images):
    training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                preprocessing_function=pipeline.keras_preprocess_func()
                                )
    #training_datagen.fit(training_datapair.x) # don't need to call fit(); see documentation
    training_generator = training_datagen.flow(x=images.x,
                                               y=images.y,
                                               batch_size=pipeline.batch_size,
                                               shuffle=True)

    return training_generator


operator_dict[build_generator] = {"module": ImageGenerator_MainNode,
                                   "inputs": [Augmentor.Pipeline],
                                   "output": Augmentor.Pipeline,
                                   "args": []
                                  }



### Primitives - Train Graph

def train_graph(graph, generator):
    history = graph.fit(x=generator,
                                       epochs=block_def.epochs,
                                       verbose=2, # TODO set to 0 or 2 after done debugging
                                       callbacks=None,
                                       validation_data=None,
                                       shuffle=True,
                                       steps_per_epoch=pipeline.num_images//pipeline.batch_size, # TODO
                                       validation_steps=None,
                                       max_queue_size=10,
                                       workers=1,
                                       use_multiprocessing=False,
                                      )
    tf.keras.backend.clear_session()

    return train_graph


operator_dict[close_graph] = {"module": TrainGraph_MainNode,
                                   "inputs": [Augmentor.Pipeline],
                                   "output": Augmentor.Pipeline,
                                   "args": []
                                  }



### Primitives - Score

def score_model(history):
    return [-1*history.history['val_accuracy'][-1], -1*history.history['val_precision'][-1], -1*history.history['val_recall'][-1]]


operator_dict[score_model] = {"module": ScoreGraph_MainNode,
                                   "inputs": [Augmentor.Pipeline],
                                   "output": Augmentor.Pipeline,
                                   "args": []
                                  }



### Compile
module_dict = {}
for method, method_dict in operator_dict.items():
    module = method_dict.module
    if module in module_dict:
        module_dict[module] = [method]
    else:
        module_dict[module].append(method)



####################################################################################################
### Create Data

class MyImages():
    def __init__(self, num_images, batch_size, image_shape=(256, 256, 3)):
        self.num_images = num_images
        self.batch_size = batch_size
        self.num_batches = num_images//batch_size
        self.image_shape = image_shape
        self.x = np.random.randint(0, 256, (num_images,)+image_shape)
        self.y = np.random.randint(0, 2, (num_images,))



class MyPipeline():
    def __init__(self, images)
        self.pipeline = Augmentor.Pipeline()
        self.num_images = images.num_images
        self.batch_size = images.batch_size
        self.num_batches = images.num_batches
        self.image_shape = images.image_shape


data_images = MyImages(num_images=200, batch_size=5)
data_pipeline = MyPipeline(data_images)



####################################################################################################
### Genome

class Genome():
    def __init__(self, input_types, num_mains, output_types, module_dict):
        self.is_dead = False
        self.num_inputs = len(input_types)
        self.num_mains = num_mains
        self.num_outputs = len(output_types)
        self.num_nodes = num_inputs + num_mains + num_outputs

        genome = []
        for _ in range(self.num_main):
            node_dict = {'module': None,
                         'value': None,
                         'function': None,
                         'inputs': []}
            genome.append(node_dict)

        for output_type in output_types:
            node_dict = {'module': output_type,
                         'inputs': []}
            genome.append(node_dict)

        for input_type in reversed(input_types):
            node_dict = {'module': input_type,
                         'value': None}
            genome.append(node_dict)

        self.genome = genome


    def match_modules(self, ith_node, module_dict, operator_dict):
        '''
        if it's a main node:
            * randomly pick operators
            * list of available modules to pick from
            * for each operator, see if it's expected modules exist

        if output node:
            * we know what module we are looking for
            * list of avaible modules to pick from
        '''
        success = False

        if ith_node < self.num_main:
            # ith node is a main node
            previous_node_modules = []
            for node in self.genome[:ith_node]:
                previous_node_modules.append(node['module'])

            # get operator choices
            operators = list(operator_dict.keys())
            random_operators = np.random.choice(operators, size=len(operators), replace=False)

            for operator in random_operators:
                operator_module_instance = operator_dict[operator]['module']()
                expected_inputs = operator_module_instance.expected_inputs

                inputs = [None]*len(expected_inputs)
                for ith_input, expected_input_tuple in enumerate(expected_inputs):
                    for possible_input_type in np.random.choice(expected_input_tuple, size=len(expected_input_tuple), replace=False)
                        if possible_input_type in random_operators:
                            inputs[ith_input] = np.where[possible_input_type=random_operators][0][0]
                            break

                if None in inputs:
                    print("failed to find match with given operator")
                    continue
                else:
                    self.genome[ith_node]['inputs'] = inputs
                    self.genome[ith_node]['function'] = operator
                    self.genome[ith_node]['module'] = operator_dict[operator]['module']()
                    success = True
                    break
                
        else:
            # assume output node
            previous_node_modules = []
            for node in self.genome[:self.num_main]:
                previous_node_modules.append(node['module'])

            # module we want
            module_instance = self.genome[ith_node]['module']()
            expected_inputs = module_instance.expected_inputs

            inputs = []
            for ith_input, expected_input_tuple in enumerate(expected_inputs):
                for possible_input_type in np.random.choice(expected_input_tuple, size=len(expected_input_tuple), replace=False)
                    if possible_input_type in previous_node_modules:
                        inputs[ith_input] = np.where[possible_input_type=previous_node_modules][0][0]
                        break

            if None in inputs:
                print("failed to find match with given output")
            else:
                self.genome[ith_node]['inputs'] = inputs
                success = True

        if not success:
            self.is_dead = True
                import pdb; pdb.set_trace()


    def build_genome(self, module_dict, operator_dict):
        '''
        genome should already have input and output modules filled in
        '''
        # main nodes
        for ith_node in range(self.num_main+self.num_outputs):
            self.match_modules(ith_node, module_dict, operator_dict)
            if self.is_dead:
                print("FAILED")
                break






