# operators
# for args, see arguments.py

import numpy as np
import tensorflow as tf
import cv2
import skimage as sk
from skimage import filters, exposure
import logging
# from tflearn.layers.conv import global_avg_pool

# dictionary to define data types for all nodes and operators
operDict = {
    "output": [np.ndarray],
    "input": [np.ndarray]
}

def add_tensors(a,b):
    return tf.add(a,b)
operDict[add_tensors] = {"inputs": [tf.Tensor, tf.Tensor],
						"outputs": tf.Tensor,
						"args": [],
                        "include_labels": False}

def sub_fa2a(a,b):
    return np.subtract(a,b)

def sub_tensors(a,b):
    return tf.subtract(a,b)
operDict[sub_tensors] = {"inputs": [tf.Tensor, tf.Tensor],
						"outputs": tf.Tensor,
						"args": [],
                        "include_labels": False}
def mult_tensors(a,b):
    return tf.multiply(a,b)
operDict[mult_tensors] = {"inputs": [tf.Tensor, tf.Tensor],
						"outputs": tf.Tensor,
						"args": [],
                        "include_labels": False}

########################## TENSORFLOW OPERATORS ###############################
"""
OpenCV methods
    1. GaussianBlur
"""
# May need to adjust the kernel size to 3? Please see here:
# https://dsp.stackexchange.com/questions/23460/how-to-calculate-gaussian-kernel-for-a-small-support-size
def gassuian_blur(input, kernel_size=5):
    output = []
    for im in input:
        output.append(cv2.GaussianBlur(im,(kernel_size,kernel_size),0))

    return np.array(output)

operDict[gassuian_blur] = {"inputs": [np.ndarray],
			   "outputs": np.ndarray,
			   "args": ['argKernelSize'],
                           "include_labels": True}


"""
Normalization methods
"""
def ceil_greyscale_norm(input):
    return input/255

operDict[ceil_greyscale_norm] = {"inputs": [np.ndarray],
                            "outputs": np.ndarray,
                            "args": [],
                            "include_labels": False
                        }

def apply_augmentation(input, labels,  percentage, function, *args):
    """
    This function is the general wrapper for data augmentation methods
    :param input: list of training x data
    :param labels: list of training y data
    :param function: function to apply
    :param percentage: percentage of the dataset to apply function to
    :param args:
    :return: augmented dataset and labels
    """
    sampleSize =  int(len(input) * percentage)
    sample_idxs = np.random.choice(len(input), sampleSize)
    output = []
    outputLabels = []
    for idx in sample_idxs:
        image_array = input[idx]
        label = labels[idx]
        # pick a random degree of rotation between 25% on the left and 25% on the right
        output.append(function(image_array, *args))
        outputLabels.append(label)
    output = np.array(output)
    outputLabels = np.array(outputLabels)
    return np.append(input, output, axis = 0), np.append(labels, outputLabels, axis = 0)

"""
see https://gist.github.com/tomahim/9ef72befd43f5c106e592425453cb6ae
for data augmentation ideas
"""
def random_rotation(input, labels, percentage = .25, rotRange = 25): #add random_degree to the arguments
    function = lambda x: sk.transform.rotate(x, np.random.uniform(-rotRange, rotRange))
    return apply_augmentation(input, labels, percentage, function)

operDict[random_rotation] = {"inputs": [np.ndarray],
                            "outputs": np.ndarray,
                            "args": ['percentage', 'rotRange'],
                            "include_labels": True}


def random_noise(input, labels, percentage = .25):
    function = sk.util.random_noise
    return apply_augmentation(input, labels, percentage, function)
    # add random noise to the image

operDict[random_noise] = {"inputs": [np.ndarray],
                            "outputs": np.ndarray,
                            "args": ['percentage'],
                            "include_labels": True}
"""
see https://gist.github.com/tomahim/9ef72befd43f5c106e592425453cb6ae
for data augmentation ideas
"""
def random_horizontal_flip(input, labels, percentage = .25): #add random_degree to the arguments
    function = lambda x: x[:, ::-1]
    return apply_augmentation(input, labels, percentage, function)

operDict[random_horizontal_flip] = {"inputs": [np.ndarray],
                            "outputs": np.ndarray,
                            "args": ['percentage'],
                            "include_labels": True}

def add_gausian_noise_to_one_image(image_array):
    mean = 0
    var = 0.1
    sigma = var**0.5
    row, col, ch = image_array.shape
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    gaussian_img = image_array + gauss
    return gaussian_img

"""
see https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9#2f4a
https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
Add gausian noise to change lighting condition
for data augmentation
"""
def add_gaussian_noise(input, labels, percentage = .25): #add random_degree to the arguments
    function = add_gausian_noise_to_one_image
    return apply_augmentation(input, labels, percentage, function)

operDict[add_gaussian_noise] = {"inputs": [np.ndarray],
                            "outputs": np.ndarray,
                            "args": ['percentage'],
                            "include_labels": True}

def add_salt_pepper_noise_to_one_image(image):
    image_array = image.copy()
    salt_vs_pepper = 0.2
    amount = 0.004
    row, col, _ = image_array.shape
    num_salt = np.ceil(amount * image_array.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * image_array.size * (1.0 - salt_vs_pepper))
    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_array.shape]
    image_array[coords[0], coords[1], :] = 1
    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_array.shape]
    image_array[coords[0], coords[1], :] = 0
    return image_array

"""
see https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9#2f4a
Add salt and pepper noise
for data augmentation
"""
def add_salt_pepper_noise(input, labels, percentage = .25): #add random_degree to the arguments
    function = add_salt_pepper_noise_to_one_image
    return apply_augmentation(input, labels, percentage, function)

operDict[add_salt_pepper_noise] = {"inputs": [np.ndarray],
                            "outputs": np.ndarray,
                            "args": ['percentage'],
                            "include_labels": True}

"""histogram equalization.
See: https://scikit-image.org/docs/0.13.x/api/skimage.exposure.html#skimage.exposure.equalize_hist
"""
def add_hist_equal(input, labels, percentage = .25):
    nbins = 32 # turn this into a param?
    function = lambda img: exposure.equalize_hist(img, nbins = nbins)
    return apply_augmentation(input, labels, percentage, function)

operDict[add_hist_equal] = {"inputs": [np.ndarray],
                            "outputs": np.ndarray,
                            "args": ['percentage'],
                            "include_labels": True}

"""
LAYERS
    1. Input
    2. Convolutional
    3. Max Pooling
    4. Average Pooling
    5. Global Average Pooling
    6. Dense
    7. Identity Layer
    8. Fractional Max Pooling
    9. Fractional Avg Pooling
"""

def input_layer(input_tensor):
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    #return tf.convert_to_tensor(data, tf.float16)
    return tf.reshape(input_tensor, data_shape)

operDict[input_layer] = {"inputs": [tf.Tensor],
			 "outputs": tf.Tensor,
			 "args": [],
             "include_labels": False}

def conv_layer(input_tensor, filters=64, kernel_size=3, activation = tf.nn.relu):
    kernel_size = (kernel_size, kernel_size)
    # Convolutional Layer
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    return tf.layers.conv2d(inputs=input_tensor, filters=filters, \
        kernel_size=kernel_size, padding="same", activation = activation, data_format = "channels_last")

operDict[conv_layer] = {"inputs": [tf.Tensor],
                            "args": ["argPow2", "argFilterSize", "activation"],
                            "outputs": tf.Tensor,
                            "name": 'convLayer',
                            "include_labels": False
                        }

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
                            "name": 'maxPoolLayer',
                            "include_labels": False}

def avg_pool_layer(input_tensor):
#    avg_pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="valid")(input_tensor)
    return tf.layers.average_pooling2d(inputs=input_tensor, pool_size=[2,2], strides=2)


operDict[avg_pool_layer] = {"inputs": [tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'avg_pool_layer',
                            "include_labels": False}

def global_avg_pool_layer(input_tensor):
    return avg_pool_layer(input_tensor)

def dense_layer(input_tensor, num_units=128, activation = tf.nn.relu):
    # Flatten tensor into a batch of vectors
    pool2_flat = tf.layers.Flatten()(input_tensor)
    # Densely connected layer with 1024 neurons
    logits = tf.layers.dense(inputs=pool2_flat, units=num_units, activation = activation)
    return logits

operDict[dense_layer] = {"inputs": [tf.Tensor],
                         "args": ["argPow2", "activation"],
                         "outputs": tf.Tensor,
                         "name": 'denseLayer',
                         "include_labels": False}

def identity_layer(input_tensor):
    print('before identity: ', input_tensor)
    output_tensor = tf.identity(input_tensor)
    print('after identity: ', output_tensor)
    return output_tensor

operDict[identity_layer] = {"inputs": [tf.Tensor],
                         "args": [],
                         "outputs": tf.Tensor,
                         "name": 'identityLayer',
                         "include_labels": False}

# TODO: see error PACE, start here
def fractional_max_pool(input_tensor, pool_height = 2.0, pool_width = 2.0):
    if input_tensor.shape[1].value == 1:
        return input_tensor
    pooling_ratio = [1.0, pool_height, pool_width, 1.0]      # see args.py for mutation limits
    pseudo_random = True        # true random underfits when combined with data augmentation and/or dropout
    overlapping = True          # overlapping pooling regions work better, according to 2015 Ben Graham paper
    # returns a tuple of Tensor objects (output, row_pooling_sequence, col_pooling_sequence
    return tf.nn.fractional_max_pool(input_tensor, pooling_ratio, pseudo_random, overlapping)[0]

operDict[fractional_max_pool] = {"inputs": [tf.Tensor],
                            "args": ['argPoolHeight', 'argPoolWidth'],
                            "outputs": tf.Tensor,
                            "name": 'fractional_max_pool',
                            "include_labels": False}

def fractional_avg_pool(input_tensor, pool_height = 2.0, pool_width = 2.0):
    if input_tensor.shape[1].value == 1:
        return input_tensor
    pooling_ratio = [1.0, pool_height, pool_width, 1.0]
    pseudo_random = True
    overlapping = True
    # returns a tuple of Tensor objects (output, row_pooling_sequence, col_pooling_sequence)
    return tf.nn.fractional_avg_pool(input_tensor, pooling_ratio, pseudo_random, overlapping)[0]

operDict[fractional_avg_pool] = {"inputs": [tf.Tensor],
                            "args": ['argPoolHeight', 'argPoolWidth'],
                            "outputs": tf.Tensor,
                            "name": 'fractional_avg_pool',
                            "include_labels": False}


def dropout_layer(input_tensor, rate = 0.2):
    #if input_tensor.shape[1].value == 1:
     #   return input_tensor
    # returns a tuple of Tensor objects (output, rate)
    return tf.nn.dropout(input_tensor, rate)


operDict[dropout_layer] = {"inputs": [tf.Tensor],
                            "args": ['percentage'],
                            "outputs": tf.Tensor,
                            "name": 'dropout_layer',
                            "include_labels": False}

def flatten_layer(input_tensor):
    #if input_tensor.shape[1].value == 1:
     #   return input_tensor
    # returns a tuple of Tensor objects (output, rate)
    return tf.layers.flatten(input_tensor)


operDict[flatten_layer] = {"inputs": [tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'flatten_layer',
                            "include_labels": False}
"""
FUNCTIONS
    1. Batch Normalization
    2. ReLu Activation
    3. Concatenation
    4. Summation
    5. Sigmoid
"""

def batch_normalization_func(input_tensor):
    # Batch Normalization
    return tf.layers.batch_normalization(input_tensor, training=True)

operDict[batch_normalization_func] = {"inputs": [tf.Tensor],
                                      "args": [],
                                      "outputs": tf.Tensor,
                                      "name": "batch_normalization",
                                      "include_labels": False}


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
                            "name": 'concatFunc',
                            "include_labels": False}

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
                            "name": 'sumFunc',
                            "include_labels": False}

def subtract_func(data1, data2):
    # Element-wise subtraction of two feature maps, channel by channel.
    # If one feature map is larger, we downsample it using max pooling
    # If one feature map has more channels, we increase its size using zero padding
    return sum_func(data1, -data2)
operDict[subtract_func] = {"inputs": [tf.Tensor, tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'subtractFunc',
                            "include_labels": False}

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
    return relu_func(batch_normalization_func(conv_layer(input_tensor, filters, kernel_size)))

operDict[conv_block] = {"inputs": [tf.Tensor],
                        "args": ["argPow2", "argFilterSize"],
                        "outputs": tf.Tensor,
                        "name": 'convBlock',
                        "include_labels": False}

def res_block(input_tensor, number1=64, size1=3, number2=128, size2=3):
    # size1 = (size1, size1)
    # size2 = (size2, size2)
    # Returns the result of one ResBlock (ConvBlock + convolutional layer + batch normalization + summation + ReLu)
    return relu_func(sum_func(batch_normalization_func(conv_layer(conv_block(input_tensor, \
        number1, size1), number2, size2)), input_tensor))

operDict[res_block] = {"inputs": [tf.Tensor],
                       "args": ["argPow2", "argFilterSize", "argPow2", "argFilterSize"],
                       "outputs": tf.Tensor,
                       "name": 'resBlock',
                       "include_labels": False}

def sqeeze_excitation_block(input_tensor):
    return sigmoid_func(dense_layer(relu_func(dense_layer(global_avg_pool_layer(input_tensor)))))

operDict[sqeeze_excitation_block] = {"inputs": [tf.Tensor],
                            "args": [],
                            "outputs": tf.Tensor,
                            "name": 'squeezeExcitationBlock',
                            "include_labels": False}

def identity_block(input_tensor, filters=64, kernel_size=3):
    # kernel_size = (kernel_size, kernel_size)
    return conv_block(conv_block(conv_block(input_tensor, filters, kernel_size)))

operDict[identity_block] = {"inputs": [tf.Tensor],
                            "args": ["argPow2", "argFilterSize"],
                            "outputs": tf.Tensor,
                            "name": 'identityBlock',
                            "include_labels": False}
