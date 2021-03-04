'''
root/codes/block_definitions/utilities/operators...

Overview:
Here we are relying on the Augmentor module to handle the 'pipeline' of our data or image processing.
Augmentor has a super simple way to do batching, and on top of that add in data preprocessing and augmentation.
Here we assume that the input and output is an Augmentor.Pipeline object and each primitive changes or adds an attribute
to that object which will eventually change how data is read in to our neural network.
All of these primitives will be Augmentor.Operations classes and we will use Augmentor.Pipeline.add_operation(method)
to add them
https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.add_operation
https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Operations.Operation

will likely be opencv

Rules:
Since we are manipulating an object instead of actually evaluating something, I decided to deepcopy the object
before returning it. In hindsight this shouldn't make a difference to the final result but it could make debugging
easier if the output from each node ends up being different things instead of different variables pointing to the same object.

Probability should always be set to 1 for preprocessing methods.
'''

### packages
import Augmentor
import numpy as np
import cv2
import functools
from copy import deepcopy
import pdb
from PIL import Image, ImageOps

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.block_definitions.utilities import argument_types
from codes.utilities.custom_logging import ezLogging
from data.data_tools import ezData


### init dict
operator_dict = {}


### decorator to convert PIL -> np; do cv2 operation; np -> PIL
def cv2_Augmentor_decorator(func):
    '''
    https://augmentor.readthedocs.io/en/master/userguide/extend.html

    Augmentor.Pipeline uses PIL.Images, and cv2 expects np.ndarray,
    so we have to convert from PIL to np, run func(), then convert np to PIL
    '''
    @functools.wraps(func)
    def wrapper_do(PIL_image):
        #print("inside %s: type %s, max %s" % (func.__name__, type(PIL_image), np.array(PIL_image).max()))
        np_image = np.array(PIL_image).astype('uint8')
        np_image = func(np_image)
        PIL_image = Augmentor.Operations.Image.fromarray(np_image) #in Augmentor.Operations they do `from PIL import Image`
        return PIL_image
    return wrapper_do



class Equalize(Augmentor.Operations.Operation):
    '''
    we're going to follow the syntax in the other Operations given in the doc
    https://augmentor.readthedocs.io/en/master/_modules/Augmentor/Operations.html

    This is our reqplacement for Normalize...since Augmentor uses PIL.Images which
    are uint8 datatypes, they can't be normalized. Next best thing is to use Equalize
    which is a PIL.Image method so we don't have to use the cv2_Augmentor_decorator
    '''
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability=1):
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        super().__init__(probability=probability)

    # Your class must implement the perform_operation method:
    def perform_operation(self, images):
        '''
        here is what the documentation says what images will be...
        images: List containing PIL.Image object(s)
        
        they like to have a do() method and then a for loop applying do() to each image
        
        NOTE here we assume that the image maxes out at 255
        '''
        def do(image):
            channels = image.split()
            mod_image = Image.merge('RGB', [ImageOps.equalize(channel)  for channel in channels[:3]])
            return mod_image
        
        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        
        return augmented_images


def equalize(ez_augmentor):
    ez_augmentor.pipeline.add_operation(Equalize())
    return ez_augmentor

operator_dict[equalize] = {"inputs": [ezData.ezData_Augmentor],
                            "output": ezData.ezData_Augmentor,
                            "args": [] # TODO: no argument because we always want prob=1 right?
                           }



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
        @cv2_Augmentor_decorator
        def do(image):
            return cv2.boxFilter(image, ddepth=-1, ksize=self.kernel, normalize=self.normalize)
        
        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        
        return augmented_images


def blur(ez_augmentor, kernel_size, normalize):
    ez_augmentor.pipeline.add_operation(Blur(kernel_size, normalize))
    return ez_augmentor


operator_dict[blur] = {"inputs": [ezData.ezData_Augmentor],
                       "output": ezData.ezData_Augmentor,
                       "args": [argument_types.ArgumentType_FilterSize,
                                argument_types.ArgumentType_Bool]
                      }



class GaussianBlur(Augmentor.Operations.Operation):
    '''
    https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
    '''
    def __init__(self, kernel_size=5, sigma_x=1, sigma_y=1, probability=1):
        super().__init__(probability=probability)
        self.kernel = (kernel_size, kernel_size)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
    
    def perform_operation(self, images):
        @cv2_Augmentor_decorator
        def do(image):
            return cv2.GaussianBlur(image, ksize=self.kernel, sigmaX = self.sigma_x, sigmaY = self.sigma_y)
        
        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        
        return augmented_images


def gaussian_blur(ez_augmentor, kernel_size, sigma_x, sigma_y):
    ez_augmentor.pipeline.add_operation(GaussianBlur(kernel_size, sigma_x, sigma_y))
    return ez_augmentor


operator_dict[gaussian_blur] = {"inputs": [ezData.ezData_Augmentor],
                                "output": ezData.ezData_Augmentor,
                                "args": [argument_types.ArgumentType_FilterSize, argument_types.ArgumentType_Float0to10, argument_types.ArgumentType_Float0to10]
                               }



class MedianBlur(Augmentor.Operations.Operation):
    '''
    https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9
    
    doc says that kernel can't be 1...going to make this true for all preprocessing so it's easier
    oh and all have to be positive + odd.
    '''
    def __init__(self, kernel_size=5, probability=1):
        super().__init__(probability=probability)
        self.kernel = kernel_size # in this case, use an int not a tuple
    
    def perform_operation(self, images):
        @cv2_Augmentor_decorator
        def do(image):
            return cv2.medianBlur(image, ksize=self.kernel)
        
        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        
        return augmented_images


def median_blur(ez_augmentor, kernel_size):
    ez_augmentor.pipeline.add_operation(MedianBlur(kernel_size))
    return ez_augmentor


operator_dict[median_blur] = {"inputs": [ezData.ezData_Augmentor],
                              "output": ezData.ezData_Augmentor,
                              "args": [argument_types.ArgumentType_FilterSize]
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
        @cv2_Augmentor_decorator
        def do(image):
            return cv2.bilateralFilter(image, d=self.d, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)
        
        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        
        return augmented_images


def bilateral_filter(ez_augmentor, d, sigma_color, sigma_space):
    ez_augmentor.pipeline.add_operation(BilateralFilter(d, sigma_color, sigma_space))
    return ez_augmentor


operator_dict[bilateral_filter] = {"inputs": [ezData.ezData_Augmentor],
                                   "output": ezData.ezData_Augmentor,
                                   "args": [argument_types.ArgumentType_Int1to10,
                                            argument_types.ArgumentType_Float0to100,
                                            argument_types.ArgumentType_Float0to100]
                                  }



class Thresholding(Augmentor.Operations.Operation):
    '''
    https://docs.opencv.org/3.4.0/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
    
    going to pass a (0-1] float and multiply that by 255 to get the threshold
    '''
    def __init__(self, thresh, maxval=255, probability=1):
        super().__init__(probability=probability)
        self.maxval = maxval
        self.thresh = self.maxval*thresh
    
    def perform_operation(self, images):
        @cv2_Augmentor_decorator
        def do(image):
            if len(image.shape) == 2 or(len(image.shape) == 3 and image.shape[2] == 1):
                _, dst = cv2.threshold(src=image, thresh=self.thresh, maxval=self.maxval, type=cv2.THRESH_BINARY)
                return dst
            return image
        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        
        return augmented_images


def thresholding(ez_augmentor, threshold):
    ez_augmentor.pipeline.add_operation(Thresholding(threshold))
    return ez_augmentor


operator_dict[thresholding] = {"inputs": [ezData.ezData_Augmentor],
                               "output": ezData.ezData_Augmentor,
                               "args": [argument_types.ArgumentType_LimitedFloat0to1]
                              }
                              
'''
class AdaptiveThreshold(Augmentor.Operations.Operation):
    ''
    https://docs.opencv.org/3.4.0/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
    ''
    def __init__(self, ith_adaptive_method, ith_threshold_type, blockSize, C, maxValue=255, probability=1):
        super().__init__(probability=probability)
        self.maxValue = maxValue
        self.blockSize = blockSize
        self.C = C
        
        adaptive_methods = [cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C]
        ith_adaptive_method = ith_adaptive_method % len(adaptive_methods)
        self.adaptiveMethod = adaptive_methods[ith_adaptive_method]
        
        threshold_types = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]
        ith_threshold_type = ith_threshold_type % len(threshold_types)
        self.thresholdType = threshold_types[ith_threshold_type]
    
    def perform_operation(self, images):
        @cv2_Augmentor_decorator
        def do(image):
            return cv2.adaptiveThreshold(image,
                                         maxValue=self.maxValue,
                                         adaptiveMethod=self.adaptiveMethod,
                                         thresholdType=self.thresholdType,
                                         blockSize=self.blockSize,
                                         C=self.C)
        
        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        
        return augmented_images


def adaptive_threshold(ez_augmentor, ith_adaptive_method, ith_threshold_type, blockSize, C):
    ez_augmentor.pipeline.add_operation(AdaptiveThreshold(ith_adaptive_method, ith_threshold_type, blockSize, C))
    return ez_augmentor


operator_dict[adaptive_threshold] = {"inputs": [ezData.ezData_Augmentor],
                                     "output": ezData.ezData_Augmentor,
                                     "args": [argument_types.ArgumentType_Int0to25,
                                              argument_types.ArgumentType_Int0to25,
                                              argument_types.ArgumentType_FilterSize,
                                              argument_types.ArgumentType_Float0to100]
                                    }'''


''' only good for greyscale images not 3 channel
class OtsuThresholding(Augmentor.Operations.Operation):
    ''
    https://docs.opencv.org/3.4.0/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
    
    going to pass a (0-1] float and multiply that by 255 to get the threshold
    ''
    def __init__(self, probability=1):
        super().__init__(probability=probability)
    
    def perform_operation(self, images):
        @cv2_Augmentor_decorator
        def do(image):
            import pdb; pdb.set_trace()
            retval, dst = cv2.threshold(src=image,
                                        thresh=0,
                                        maxval=255,
                                        type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return dst
        
        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        
        return augmented_images


def otsu_thresholding(ez_augmentor):
    ez_augmentor.pipeline.add_operation(OtsuThresholding())
    return ez_augmentor


operator_dict[otsu_thresholding] = {"inputs": [ezData.ezData_Augmentor],
                                    "output": ezData.ezData_Augmentor,
                                    "args": []
                                   }'''


