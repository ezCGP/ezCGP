'''
removed normalize() from preprocessing primitives so now I want to play with how to normalize with the tf.keras
ImageDataGen. but then another question is, what happens when we pass in uint8 (0,256) values vs np.float (0,1)?
Augmentor.Pipeline.keras_preprocessing_ftn() expects float values between (0,1), and then converts it to PIL.Image
but never converts back to float btwn (0,1). I want to verify the order of events.
'''

### packages
import os
import numpy as np
from copy import deepcopy
import pdb
import Augmentor
import tensorflow as tf

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from misc import play_with_preprocessing as pwp


def experiment(images_asint=True,
               wPreprocessing=True,
               **kwargs):
    pipeline = pwp.create_pipeline()
    images, labels = pwp.create_fake_data(asint=images_asint)

    if wPreprocessing:
        pp_ftn = pipeline.keras_preprocess_func()
    else:
        pp_ftn = None
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                preprocessing_function=pp_ftn,
                                **kwargs
                                )

    generator = datagen.flow(x=images,
                             y=labels,
                             batch_size=5,
                             shuffle=False)

    return images, generator

for asint in [True, False]:
    for wPreprocess in [True, False]:
        for normalized in [True, False]:
            print("Original data as uint8 btwn 0 and 255: %s" % asint)
            print("ImageDataGenerator with pipeline: %s" % wPreprocess)
            print("Normalized: %s" % normalized)
            normalized_dict = {}
            if normalized:
                normalized_dict['featurewise_center'] = True
                normalized_dict['featurewise_std_normalization'] = True
            images, generator = experiment(images_asint=asint, **normalized_dict)
        
            image_batch, label_batch = generator.next()
            print("Starting images: %s w/Max %s" % (images.dtype.__str__(), images.max()))
            print("Final images: %s w/Max %s" % (image_batch.dtype.__str__(), image_batch.max()))
            print("")

