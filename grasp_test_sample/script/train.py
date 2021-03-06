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
import csv
import sys
sys.path.insert(1, "/home/allen/.local/lib/python3.5/site-packages/")
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pandas as pd
from functools import partial

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# def pd_read_csv(filelist):
 
#     data_frame = pd.read_csv(filelist, sep=",")
#     data1 = []
#     data2 = []
   
#     for index in data_frame.index:
#         data_row = data_frame.loc[index]
#         data1.append(data_row["x"])
#         data2.append(data_row["two"])
#     return (data1,data2)
TRAIN_DATA_URL = "http://home/allen/dl_grasp/src/rs_d435i/Test/123.csv"
file_path="/home/allen/dl_grasp/src/grasp_test_sample/data/square_data.csv"
#TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"

# def normalize_number_data(data):
#     data=data/255
#     return data

# def turn_batch(dataset):
#     for batch in dataset.take(10):
#         for key,value in batch.items():
#             print("{:20s}: {}".format(key,value.numpy()))
#         print('-----------------')

def show_batch(dataset):
    for batch in dataset.take(10):
        for key,value in batch.items():
            print("{:20s}: {}".format(key,value.numpy()))
        print('-----------------')

def get_dataset(file_path):
    #LABEL_COLUM='target_x'
    dataset = tf.data.experimental.make_csv_dataset(
    file_path,
    batch_size=1,
    field_delim=',',
    column_names=['image_path','target_x','target_y'],
    column_defaults=[tf.string, tf.float32,tf.float32],
    #label_name=LABEL_COLUM,
    na_value="?",
    num_epochs=100,
    shuffle_seed=True,
    ignore_errors=True)
    return dataset

def main():
    print('train_model_123')
    aa=get_dataset(file_path)
    print('train_model_456')
    show_batch(aa)
    print('train_model_789')
    examples= next(iter(aa))
    print("EXAMPLES: \n", examples, "\n")


if __name__ == "__main__":
    main()