#!/usr/bin/env python3
import rospy
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
import csv
import pandas as pd

def read_csv():
    print('sdfasd')
    data_frame =pd.read_csv("/home/allen/dl_grasp/src/grasp_test_sample/data/data.csv",sep=",")
    
def main():
    read_csv()
    
if __name__ == "__main__":

    main()



