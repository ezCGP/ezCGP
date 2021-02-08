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


pipeline = pwp.create_pipeline()
images, labels = pwp.create_fake_data(asint=False)


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                preprocessing_function=pipeline.keras_preprocess_func()
                                )

generator = datagen.flow(x=images,
                         y=labels,
                         batch_size=5,
                         shuffle=True)

for ith_batch, batch in enumerate(generator.next()):
    print("%ith batch" % ith_batch)
    pdb.set_trace()