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
'''

### packages
import Augmentor
import numpy as np
import cv2
from copy import deepcopy

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

### absolute imports wrt root
from codes.block_definitions.utilities import argument_types
from codes.utilities.custom_logging import ezLogging


### init dict
operator_dict = {}



class Normalize(Augmentor.Operations.Operation):
    '''
    we're going to follow the syntax in the other Operations given in the doc
    https://augmentor.readthedocs.io/en/master/_modules/Augmentor/Operations.html
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
            mod_image = np.asarray(image) / 255.0
            return mod_image
        
        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        
        return augmented_images


def normalize(pipeline):
    pipeline.add_operation(Normalize())
    return deepcopy(pipeline)


operator_dict[normalize] = {"inputs": [Augmentor.Pipeline],
                            "output": Augmentor.Pipeline,
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
        def do(image):
            return cv2.blur(image, ksize=self.kernel, normalize=self.normalize)
        
        augmented_images = []
        for image in images:
            augmented.append(do(image))
        
        return augmented_images


def blur(pipeline):
    pipeline.add_operation(Blur())
    return deepcopy(pipeline)


operator_dict[blur] = {"inputs": [Augmentor.Pipeline],
                       "output": Augmentor.Pipeline,
                       "args": [argument_types.ArgumentType_FilterSize,
                                argument_types.ArgumentType_Bool]
                      }



class GaussianBlur(Augmentor.Operations.Operation):
    '''
    https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
    '''
    def __init__(self, kernel_size=5, probability=1):
        super().__init__(probability=probability)
        self.kernel = (kernel_size, kernel_size)
    
    def perform_operation(self, images):
        def do(image):
            return cv2.GaussianBlur(image, ksize=self.kernel)
        
        augmented_images = []
        for image in images:
            augmented.append(do(image))
        
        return augmented_images


def gaussian_blur(pipeline):
    pipeline.add_operation(GaussianBlur())
    return deepcopy(pipeline)


operator_dict[gaussian_blur] = {"inputs": [Augmentor.Pipeline],
                                "output": Augmentor.Pipeline,
                                "args": [argument_types.ArgumentType_FilterSize]
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
        def do(image):
            return cv2.medianBlur(image, ksize=self.kernel)
        
        augmented_images = []
        for image in images:
            augmented.append(do(image))
        
        return augmented_images


def median_blur(pipeline):
    pipeline.add_operation(MedianBlur())
    return deepcopy(pipeline)


operator_dict[median_blur] = {"inputs": [Augmentor.Pipeline],
                              "output": Augmentor.Pipeline,
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
        def do(image):
            return cv2.bilateralFilter(image, d=self.d, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)
        
        augmented_images = []
        for image in images:
            augmented.append(do(image))
        
        return augmented_images


def bilateral_filter(pipeline, d, sigma_color, sigma_space):
    pipeline.add_operation(BilateralFilter(d, sigma_color, sigma_space))
    return deepcopy(pipeline)


operator_dict[bilateral_filter] = {"inputs": [Augmentor.Pipeline],
                                   "output": Augmentor.Pipeline,
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
        def do(image):
            retval, dst = cv2.threshold(image, thresh=self.thresh, maxval=self.maxval)
            return dst
        
        augmented_images = []
        for image in images:
            augmented.append(do(image))
        
        return augmented_images


def thresholding(pipeline, threshold):
    pipeline.add_operation(Thresholding(threshold))
    return deepcopy(pipeline)


operator_dict[thresholding] = {"inputs": [Augmentor.Pipeline],
                               "output": Augmentor.Pipeline,
                               "args": [argument_types.ArgumentType_LimitedFloat0to1]
                              }



class AdaptiveThreshold(Augmentor.Operations.Operation):
    '''
    https://docs.opencv.org/3.4.0/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
    
    going to pass a (0-1] float and multiply that by 255 to get the threshold
    '''
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
        def do(image):
            return cv2.adaptiveThreshold(image,
                                         maxValue=self.maxValue,
                                         adaptiveMethod=self.adaptiveMethod,
                                         thresholdType=self.thresholdType,
                                         blockSize=self.blockSize,
                                         C=self.C)
        
        augmented_images = []
        for image in images:
            augmented.append(do(image))
        
        return augmented_images


def adaptive_threshold(pipeline, ith_adaptive_method, ith_threshold_type, blockSize, C):
    pipeline.add_operation(AdaptiveThreshold(ith_adaptive_method, ith_threshold_type, blockSize, C))
    return deepcopy(pipeline)


operator_dict[adaptive_threshold] = {"inputs": [Augmentor.Pipeline],
                                     "output": Augmentor.Pipeline,
                                     "args": [argument_types.ArgumentType_LimitedFloat0to1,
                                              argument_types.ArgumentType_Int0to25,
                                              argument_types.ArgumentType_Int0to25,
                                              argument_types.ArgumentType_FilterSize,
                                              argument_types.ArgumentType_Float0to100]
                                    }



class OtsuThresholding(Augmentor.Operations.Operation):
    '''
    https://docs.opencv.org/3.4.0/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
    
    going to pass a (0-1] float and multiply that by 255 to get the threshold
    '''
    def __init__(self, probability=1):
        super().__init__(probability=probability)
    
    def perform_operation(self, images):
        def do(image):
            retval, dst = cv2.threshold(image,
                                        thresh=0,
                                        maxval=255,
                                        type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return dst
        
        augmented_images = []
        for image in images:
            augmented.append(do(image))
        
        return augmented_images


def otsu_thresholding(pipeline):
    pipeline.add_operation(OtsuThresholding())
    return deepcopy(pipeline)


operator_dict[otsu_thresholding] = {"inputs": [Augmentor.Pipeline],
                                    "output": Augmentor.Pipeline,
                                    "args": []
                                   }


