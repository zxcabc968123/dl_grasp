#!/usr/bin/env python3
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy
import tensorflow.keras.backend as kb

lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
sys.path.insert(1, "/home/allen/.local/lib/python3.5/site-packages/")
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pandas as pd
import math


def custom_loss(y_actual,y_pred):
    x_gap = tf.math.squared_difference(y_pred[:,0], y_actual[:,0])
    #y_gap = tf.math.squared_difference(y_pred[:,1], y_actual[:,1])
    loss = tf.square(y_pred[:,2]-y_actual[:,2])
    print(loss)
    return tf.math.reduce_mean(loss)


def main():
    x = tf.cast([[1, 3, 5], [7, 9, 11]],dtype=tf.float32)
    y = tf.cast([[2, 4, 6], [7, 10, 12]],dtype=tf.float32)
    print(x.ndim)
    print(y.ndim)
    #print(x[0][1])
    z = custom_loss(x,y)   
    print(z)


if __name__ == "__main__":
    main()