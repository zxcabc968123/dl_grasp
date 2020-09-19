#!/usr/bin/env python3
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.insert(1, "/home/allen/.local/lib/python3.5/site-packages/")
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')

import cv2, os, sys, getopt
import argparse
import datetime 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
train_image_path = "/home/allen/dl_grasp/src/rs_d435i/pic/depth_train/locate"
train_path = "/home/allen/dl_grasp/src/rs_d435i/pic/depth_train"
IMG_SHAPE = 150

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

def main():
    depth_image=os.listdir(train_path)
    depth_image_num = len(os.listdir(train_image_path))
    print(depth_image)
    print('num of files  {}'.format(depth_image_num))

    train_image_generator = ImageDataGenerator(rescale=1./255)
    train_data_gen = train_image_generator.flow_from_directory(batch_size=50,
                                                           directory=train_path,
                                                           shuffle=True,
                                                           target_size=(150,150), #(150,150)
                                                           class_mode='binary')
    print(len(train_data_gen))
    print(train_data_gen[0][0][0].shape)
    image=cv2.imread('/home/allen/dl_grasp/src/rs_d435i/pic/depth_test/locate/205.jpg',0)
    #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    print(image.shape)
    plt.imshow(image,cmap="gray")
    plt.show()


    sample_training_images, _ = next(train_data_gen) 
    #plotImages(sample_training_images[:15])
   





    
if __name__ == "__main__":
    
    main()

