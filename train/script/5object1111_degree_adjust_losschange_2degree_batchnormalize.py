#!/usr/bin/env python3
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5800)])
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D ,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy
import tensorflow.keras.backend as kb
from solve_cudnn_error import *

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

train_file_path = '/home/allen/dl_grasp/src/data_expend/expand_data/5object_2500-11-11_07_30_55_.csv'
test_file_path = '/home/allen/dl_grasp/src/data_expend/expand_data/5object_100_2020-11-11_07_50_16_.csv'
save_path = '/home/allen/dl_grasp/src/train/Save_net/5object_1111'
Batch_size = 5
EPOCHS = 500

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def custom_loss(y_actual,y_pred):
    x_gap = tf.square(y_pred[:,0]-y_actual[:,0])
    y_gap = tf.square(y_pred[:,1]-y_actual[:,1])
    cos_gap = tf.square(y_pred[:,2]-y_actual[:,2])
    sin_gap = tf.square(y_pred[:,3]-y_actual[:,3])

    loss = x_gap + y_gap + 1.5*cos_gap + 1.5*sin_gap

    return tf.math.sqrt(tf.math.reduce_mean(loss))

def pd_read_csv(csvFile):
 
    data_frame = pd.read_csv(csvFile, sep=",")
    data1 = []
    data2 = []
    data3 = []
    data4 = []
   
    for index in data_frame.index:
        data_row = data_frame.loc[index]
        data1.append(data_row["image_path"])
        data2.append(data_row["target_x"])
        data3.append(data_row["target_y"])
        data4.append(data_row["target_angle"])

    for i in range(len(data4)):
         #######################
        if data4[i]>90:
            data4[i] = -1*(data4[i]-90)
        elif data4[i]<=90:
            data4[i] = 90-data4[i]
    return (data1,data2,data3,data4)

def create_result_array(data2,data3,data4):

    result_array=np.zeros([len(data2),4],  dtype=float)

    for i in range(len(data2)):
        diameter = data4[i]*math.pi/180
        cos_pi = math.cos(2*diameter)
        sin_pi = math.sin(2*diameter)
        result_array[i]=[data2[i]/640,data3[i]/480,cos_pi,sin_pi]
        # print('cos : {} sin : {}'.format(cos_pi,sin_pi))
    return result_array

def create_photo_array(data1):

    photo_array= cv2.imread(data1[0],0)
    #photo_array=photo_array/255
    for i in range(len(data1)-1):
        image=cv2.imread(data1[i+1],0)
        ###
        #image = image/255
        photo_array=np.concatenate((photo_array,image))
        print(i)
    photo_array=photo_array.reshape((-1,480,640,1))
    ########normalize
    photo_array=photo_array/255
    return photo_array

def main():
    ####################   train  data_create
    data1, data2, data3, data4 = pd_read_csv(train_file_path)
    train_photo_array=create_photo_array(data1)
    train_result_array=create_result_array(data2,data3,data4)
    print('photo_array shape: {}'.format(train_photo_array.shape))
    print('result_array shape: {}'.format(train_result_array.shape))

    ####################   test   data_create
    data1, data2, data3, data4 = pd_read_csv(test_file_path)
    test_photo_array=create_photo_array(data1)
    test_result_array=create_result_array(data2,data3,data4)
    print('photo_array shape: {}'.format(test_photo_array.shape))
    print('result_array shape: {}'.format(test_result_array.shape))

    ###################   Network
    CNN=keras.Sequential()
    #add convolution layer filter 32 3*3 activation funtion relu
    CNN.add(layers.Conv2D(64,(2,2),input_shape=(480,640,1)))
    #CNN.add(layers.BatchNormalization())
    CNN.add(layers.Activation(lrelu))
    CNN.add(layers.MaxPooling2D((2,2)))

    CNN.add(layers.Conv2D(32,(2,2),strides=(2, 2)))
    #CNN.add(layers.BatchNormalization())
    CNN.add(layers.Activation(lrelu))
    CNN.add(layers.MaxPooling2D((2,2)))

    CNN.add(layers.Conv2D(32,(2,2),strides=(2, 2)))
    #CNN.add(layers.BatchNormalization())
    CNN.add(layers.Activation(lrelu))
    CNN.add(layers.MaxPooling2D((2,2)))
    

    #####Dropout
    #CNN.add(layers.Dropout(0.5))
    CNN.add(layers.Flatten())
   
    #CNN.add(layers.Dropout(0.5))
    CNN.add(layers.Dense(64))
    
    CNN.add(layers.Activation(lrelu))
    CNN.add(layers.BatchNormalization())

    #CNN.add(layers.Dropout(0.5))
    CNN.add(layers.Dense(32))
    
    CNN.add(layers.Activation(lrelu))
    CNN.add(layers.BatchNormalization())

    #CNN.add(layers.Dropout(0.5))
    CNN.add(layers.Dense(16))
    
    CNN.add(layers.Activation(lrelu))

    #CNN.add(layers.Dense(4))
    CNN.add(layers.Dense(4))

    #CNN.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.0006),metrics=['mae'])
    CNN.compile(loss=custom_loss,optimizer=tf.keras.optimizers.Adam(0.006),metrics=['mae'])
    CNN.summary()

    result=CNN.fit(train_photo_array,train_result_array,batch_size=Batch_size,epochs=EPOCHS,shuffle=True,verbose=1)

    print('save CNN')
    CNN.save(save_path, save_format='tf')

    #################### Loss/epoch
    loss = result.history['loss']
    epochs_range = range(EPOCHS)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.savefig(save_path+'/loss.png')
    plt.show()
    #####
    #score = model.evaluate(x_test, y_test, verbose=0)

    
if __name__ == "__main__":
    main()
    

