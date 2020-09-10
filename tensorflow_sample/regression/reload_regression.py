#!/usr/bin/env python3
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

export_path='/home/allen/dl_grasp/src/tensorflow_sample/regression/SaveNet'
def main():
    reload_sm_keras = tf.keras.models.load_model(export_path)
    reload_sm_keras.summary()
    print(reload_sm_keras.predict([100]))
if __name__ == "__main__":
    main()