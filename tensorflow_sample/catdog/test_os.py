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
train_image_path = "/home/allen/dl_grasp/src/rs_d435i/pic/depth_train/locate"
train_path = "/home/allen/dl_grasp/src/rs_d435i/pic/depth_train"
IMG_SHAPE = 150
def main():
    depth_image=os.listdir(train_path)
    depth_image_num = len(os.listdir(train_image_path))
    print(depth_image)
    print('num of files  {}'.format(depth_image_num))

    train_image_generator = ImageDataGenerator(rescale=1./255)
    train_data_gen = train_image_generator.flow_from_directory(batch_size=10,
                                                           directory=train_path,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)
                                                           class_mode='binary')

    
if __name__ == "__main__":
    
    main()

