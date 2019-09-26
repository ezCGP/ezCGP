# operators

import numpy as np
import tensorflow as tf
import cv2
# #from tflearn.layers.conv import global_avg_pool

# dictionary to define data types for all nodes and operators
operDict = {
    "output": [np.ndarray],
    "input": [np.ndarray]
}

def add_tensors(a,b):
    return tf.add(a,b)
operDict[add_tensors] = {"inputs": [tf.Tensor, tf.Tensor],
						"outputs": tf.Tensor,
						"args": []}

def sub_fa2a(a,b):
    return np.subtract(a,b)

def sub_tensors(a,b):
    return tf.subtract(a,b)
operDict[sub_tensors] = {"inputs": [tf.Tensor, tf.Tensor],
						"outputs": tf.Tensor,
						"args": []}
def mult_tensors(a,b):
    return tf.multiply(a,b)
operDict[mult_tensors] = {"inputs": [tf.Tensor, tf.Tensor],
						"outputs": tf.Tensor,
						"args": []}

########################## TENSORFLOW OPERATORS ###############################
"""
OpenCV methods
    1. GaussianBlur
"""
def gassuian_blur(input, kernel_size=5):
    output = []
    for im in input:
        output.append(cv2.GaussianBlur(im,(kernel_size,kernel_size),0))

    return np.array(output)

operDict[gassuian_blur] = {"inputs": [np.ndarray],
			   "outputs": np.ndarray,
			   "args": ['argKernelSize']}

"""
LAYERS
    1. Input
    2. Convolutional
    3. Max Pooling
    4. Average Pooling
    5. Global Average Pooling
    6. Dense
    7. Identity Layer
"""

def input_layer(input_tensor):
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    #return tf.convert_to_tensor(data, tf.float16)
    return tf.reshape(input_tensor, data_shape)

operDict[input_layer] = {"inputs": [tf.Tensor],
			 "outputs": tf.Tensor,
			 "args": []}

def conv_layer(input_tensor, filters=64, kernel_size=3):
    kernel_size = (kernel_size, kernel_size)
    # Convolutional Layer
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    return tf.layers.conv2d(inputs=input_tensor, filters=filters, \
        kernel_size=kernel_size, padding="same", activation=None, data_format = "channels_last")

operDict[conv_layer] = {"inputs": [tf.Tensor],
                            "args": ["argPow2", "argFilterSize"],
                            "outputs": tf.Tensor,
                            "name": 'convLayer'}

def max_pool_layer(input_tensor):
    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    if input_tensor.shape[1].value == 1:
        return input_tensor
    #max_pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="valid")(input_tensor)
    return tf.layers.max_pooling2d(inputs=input_tensor, pool_size=[2, 2], strides=2)
#
#
operDict[max_pool_layer] = {"inputs": [tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'maxPoolLayer'}

def avg_pool_layer(input_tensor):
#    avg_pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="valid")(input_tensor)
    return tf.layers.average_pooling2d(inputs=input_tensor, pool_size=[2,2], strides=2)


operDict[avg_pool_layer] = {"inputs": [tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'avgPoolLayer'}

def global_avg_pool_layer(input_tensor):
    return global_avg_pool(input_tensor)

def dense_layer(input_tensor, num_units=128):
    # Flatten tensor into a batch of vectors
    pool2_flat = tf.layers.Flatten()(input_tensor)
    # Densely connected layer with 1024 neurons
    logits = tf.layers.dense(inputs=pool2_flat, units=num_units, activation=tf.nn.relu)
    return logits

operDict[dense_layer] = {"inputs": [tf.Tensor],
                         "args": ["argPow2"],
                         "outputs": tf.Tensor,
                         "name": 'denseLayer'}

def identity_layer(input_tensor):
    print('before identity: ', input_tensor)
    output_tensor = tf.identity(input_tensor)
    print('after identity: ', output_tensor)
    return output_tensor

operDict[identity_layer] = {"inputs": [tf.Tensor],
                         "args": [],
                         "outputs": tf.Tensor,
                         "name": 'identityLayer'}

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

operDict[concat_func] = {"inputs": [tf.Tensor, tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'concatFunc'}

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

operDict[sum_func] = {"inputs": [tf.Tensor, tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'sumFunc'}

def sigmoid_func(input_tensor):
    return tf.nn.sigmoid(input_tensor)


"""
BLOCKS
    1. ConvBlock
    2. ResBlock
    3. SEBlock
"""

def conv_block(input_tensor, filters=64, kernel_size=3):
    # Returns the result of one ConvBlock (convolutional layer + batch normalization + ReLu)
    # kernel_size = (kernel_size, kernel_size)
    return relu_func(norm_func(conv_layer(input_tensor, filters, kernel_size)))

operDict[conv_block] = {"inputs": [tf.Tensor],
                            "args": ["argPow2", "argFilterSize"],
                            "outputs": tf.Tensor,
                            "name": 'convBlock'}

def res_block(input_tensor, number1=64, size1=3, number2=128, size2=3):
    # size1 = (size1, size1)
    # size2 = (size2, size2)
    # Returns the result of one ResBlock (ConvBlock + convolutional layer + batch normalization + summation + ReLu)
    return relu_func(sum_func(norm_func(conv_layer(conv_block(input_tensor, \
        number1, size1), number2, size2)), input_tensor))

operDict[res_block] = {"inputs": [tf.Tensor],
                            "args": ["argPow2", "argFilterSize", "argPow2", "argFilterSize"],
                            "outputs": tf.Tensor,
                            "name": 'resBlock'}

def sqeeze_excitation_block(input_tensor):
    return sigmoid_func(dense_layer(relu_func(dense_layer(global_avg_pool_layer(input_tensor)))))

operDict[sqeeze_excitation_block] = {"inputs": [tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'squeezeExcitationBlock'}

def identity_block(input_tensor, filters=64, kernel_size=3):
    # kernel_size = (kernel_size, kernel_size)
    return conv_block(conv_block(conv_block(input_tensor, filters, kernel_size)))

operDict[identity_block] = {"inputs": [tf.Tensor],
                            "args": ["argPow2", "argFilterSize"],
                            "outputs": tf.Tensor,
                            "name": 'identityBlock'}
