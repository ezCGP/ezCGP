# operators

import numpy as np
import tensorflow as tf
# # from tflearn.layers.conv import global_avg_pool

# dictionary to define data types for all nodes and operators
operDict = {
    "output": [np.ndarray],
    "input": [np.ndarray]
}
# #TODO this seems redundant; gotta clean this up

# def add_ff2f(a,b):
#     return np.add(a,b)
# #operDict[add_ff2f] = [[np.float64, np.float64], np.float64]
# operDict[add_ff2f] = {"inputs": [np.float64, np.float64],
# 						"outputs": np.float64,
# 						"args": []
# 						}

# def add_fa2a(a,b):
#     return np.add(a,b)
# #operDict[add_fa2a] = [[np.float64,np.ndarray], np.ndarray]
# operDict[add_fa2a] = {"inputs": [np.ndarray, np.float64],
# 						"outputs": np.ndarray,
# 						"args": []
# 						}
# """
# operDict[add_fa2a] = {"inputs": [np.ndarray],
# 						"outputs": np.ndarray,
# 						"args": [FloatSmall],
# 						"num_args": 3}
# """

# def add_aa2a(a,b):
#     return np.add(a,b)
# #operDict[add_aa2a] = [[np.ndarray,np.ndarray], np.ndarray]
# operDict[add_aa2a] = {"inputs": [np.ndarray, np.ndarray],
# 						"outputs": np.ndarray,
# 						"args": []
# 						}
# #operDict[add_aa2a] = {"inputs": [np.ndarray, np.ndarray],
# #						"outputs": np.ndarray,
# #						"args": [],
# #						"num_args": 3}

# def sub_ff2f(a,b):
#     return np.subtract(a,b)
# #operDict[sub_ff2f] = [[np.float64, np.float64], np.float64]
# operDict[sub_ff2f] = {"inputs": [np.float64, np.float64],
# 						"outputs": np.float64,
# 						"args": []
# 						}

# def sub_fa2a(a,b):
#     return np.subtract(a,b)
# #operDict[sub_fa2a] = [[np.float64,np.ndarray], np.ndarray]
# operDict[sub_fa2a] = {"inputs": [np.float64, np.ndarray],
# 						"outputs": np.ndarray,
# 						"args": []
# 						}
# #operDict[sub_fa2a] = {"inputs": [np.ndarray],
# #						"outputs": np.ndarray,
# #						"args": [FloatSmall],
# #						"num_args": 3}

# def sub_aa2a(a,b):
#     return np.subtract(a,b)
# #operDict[sub_aa2a] = [[np.ndarray,np.ndarray], np.ndarray]
# operDict[sub_aa2a] = {"inputs": [np.ndarray, np.ndarray],
# 						"outputs": np.ndarray,
# 						"args": []
# 						}
# #operDict[sub_aa2a] = {"inputs": [np.ndarray, np.ndarray],
# #						"outputs": np.ndarray,
# #						"args": [],
# #						"num_args": 3}

# def mul_ff2f(a,b):
#     return np.multiply(a,b)
# #operDict[mul_ff2f] = [[np.float64, np.float64], np.float64]
# operDict[mul_ff2f] = {"inputs": [np.float64, np.float64],
# 						"outputs": np.float64,
# 						"args": []
# 						}

# def mul_fa2a(a,b):
#     return np.multiply(a,b)
# #operDict[mul_fa2a] = [[np.float64,np.ndarray], np.ndarray]
# operDict[mul_fa2a] = {"inputs": [np.float64, np.ndarray],
# 						"outputs": np.ndarray,
# 						"args": []
# 						}
# #operDict[mul_fa2a] = {"inputs": [np.ndarray],
# #						"outputs": np.ndarray,
# #						"args": [FloatSmall],
# #						"num_args": 3}

# def mul_aa2a(a,b):
#     return np.multiply(a,b)
# #operDict[mul_aa2a] = [[np.ndarray,np.ndarray], np.ndarray]
# operDict[mul_aa2a] = {"inputs": [np.ndarray, np.ndarray],
# 						"outputs": np.ndarray,
# 						"args": []
# 						}
# #operDict[mul_aa2a] = {"inputs": [np.ndarray, np.ndarray],
# #						"outputs": np.ndarray,
# #						"args": [],
# #						"num_args": 3}

def add_tensors(a,b):
    return tf.add(a,b)
operDict[add_tensors] = {"inputs": [tf.Tensor, tf.Tensor],
						"outputs": tf.Tensor,
						"args": []
						}
def sub_tensors(a,b):
    return tf.subtract(a,b)
operDict[sub_tensors] = {"inputs": [tf.Tensor, tf.Tensor],
						"outputs": tf.Tensor,
						"args": []
						}
def mult_tensors(a,b):
    return tf.multiply(a,b)
operDict[mult_tensors] = {"inputs": [tf.Tensor, tf.Tensor],
						"outputs": tf.Tensor,
						"args": []
						}

########################## TENSORFLOW OPERATORS ###############################

# operDict = {
#     "output": [np.ndarray], # output labels
#     "input": [np.ndarray] # input data
# }

"""
LAYERS
    1. Input
    2. Convolutional
    3. Max Pooling
    4. Average Pooling
    5. Global Average Pooling
    6. Dense
"""

def input_layer(input_tensor):
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    #return tf.convert_to_tensor(data, tf.float16)
    return tf.reshape(input_tensor, data_shape)

operDict[input_layer] = {"inputs": [tf.Tensor],
					"outputs": tf.Tensor,
					"args": []
					}

def conv_layer(input_tensor, filters=64, kernel_size=(3, 3)):
    # Convolutional Layer
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    return tf.layers.conv2d(inputs=input_tensor, filters=filters, \
        kernel_size=kernel_size, padding="same", activation=None)


def max_pool_layer(input_tensor):
    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    if input_tensor.shape[1].value == 1:
        return input_tensor
    #max_pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="valid")(input_tensor)
    return tf.layers.max_pooling2d(inputs=input_tensor, pool_size=[2, 2], strides=2)
#
#
def avg_pool_layer(input_tensor):
#    avg_pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="valid")(input_tensor)
    return tf.layers.average_pooling2d(inputs=input_tensor, pool_size=[2,2], strides=2)


def global_avg_pool_layer(input_tensor):
    return global_avg_pool(input_tensor)
#
#
def dense_layer(input_tensor, num_units=128):
    # Flatten tensor into a batch of vectors
    pool2_flat = tf.layers.Flatten()(input_tensor)
    # Densely connected layer with 1024 neurons
    logits = tf.layers.dense(inputs=pool2_flat, units=num_units, activation=tf.nn.relu)
    return logits

"""
FUNCTIONS
    1. Batch Normalization
    2. ReLu Activation
    3. Concatenation
    4. Summation
    5. Sigmoid
"""


def norm_func(input_tensor):
    # Batch Normalization
    return tf.layers.batch_normalization(input_tensor, training=True)


def relu_func(input_tensor):
    # ReLu Non-linear activation function
    return tf.nn.relu(input_tensor)


def concat_func(data1, data2):
    # Concatenates two feature maps in the channel dimension
    # If one feature map is larger, we downsample it using max pooling
    if data1.shape[1].value > data2.shape[1].value:
        data1 = max_pool_layer(data1)
        return concat_func(data1, data2)
    elif data1.shape[1].value < data2.shape[1].value:
        data2 = max_pool_layer(data2)
        return concat_func(data1, data2)
    else:
        return tf.concat([data1, data2], 3)


def sum_func(data1, data2):
    # Element-wise addition of two feature maps, channel by channel.
    # If one feature map is larger, we downsample it using max pooling
    # If one feature map has more channels, we increase its size using zero padding
    if data1.shape[1].value > data2.shape[1].value:
        data1 = max_pool_layer(data1)
        return sum_func(data1, data2)
    elif data1.shape[1].value < data2.shape[1].value:
        data2 = max_pool_layer(data2)
        return sum_func(data1, data2)
    else:
        diff = data1.shape[3].value - data2.shape[3].value
        if diff > 0:
            data2 = tf.pad(data2, tf.constant([[0, 0], [0, 0], [0, 0], [0, diff]]), "CONSTANT")
        elif diff < 0:
            data1 = tf.pad(data1, tf.constant([[0, 0], [0, 0], [0, 0], [0, (-diff)]]), "CONSTANT")
        else:
            pass
        out = tf.add(data1, data2)
        return out


def sigmoid_func(input_tensor):
    return tf.nn.sigmoid(input_tensor)


"""
BLOCKS
    1. ConvBlock
    2. ResBlock
    3. SEBlock
"""

def conv_block(input_tensor, filters=64, kernel_size=(3, 3)):
    # Returns the result of one ConvBlock (convolutional layer + batch normalization + ReLu)
    return relu_func(norm_func(conv_layer(input_tensor, filters, kernel_size)))


def res_block(input_tensor, number1=64, size1=(3,3), number2=128, size2=(3,3)):
    # Returns the result of one ResBlock (ConvBlock + convolutional layer + batch normalization + summation + ReLu)
    return relu_func(sum_func(norm_func(conv_layer(conv_block(input_tensor, \
        number1, size1), number2, size2)), input_tensor))


def sqeeze_excitation_block(input_tensor):
    return sigmoid_func(dense_layer(relu_func(dense_layer(global_avg_pool_layer(input_tensor)))))


def identity_block(input_tensor, filters=64, kernel_size=(3,3)):
    return conv_block(conv_block(conv_block(input_tensor, filters, kernel_size)))


operDict[dense_layer] = {"inputs": [tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'denseLayer'}

operDict[conv_layer] = {"inputs": [tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'convLayer'}

operDict[max_pool_layer] = {"inputs": [tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'maxPoolLayer'}

operDict[avg_pool_layer] = {"inputs": [tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'avgPoolLayer'}

operDict[concat_func] = {"inputs": [tf.Tensor, tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'concatFunc'}

operDict[sum_func] = {"inputs": [tf.Tensor, tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'sumFunc'}

operDict[conv_block] = {"inputs": [tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'convBlock'}


operDict[res_block] = {"inputs": [tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'resBlock'}

operDict[sqeeze_excitation_block] = {"inputs": [tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'squeezeExcitationBlock'}

operDict[identity_block] = {"inputs": [tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'identityBlock'}

'''Commented and left until args are checked'''

# operDict[conv_block] = {"inputs": [tf.Tensor],
#                             "args": [argInt, argInt],
#                             "outputs": tf.Tensor,
#                             "name": 'convBlock'}

# operDict[res_block] = {"inputs": [tf.Tensor],
#                             "args": [argInt, argInt, argInt, argInt],
#                             "outputs": tf.Tensor,
#                             "name": 'resBlock'}
