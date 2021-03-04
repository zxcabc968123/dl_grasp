#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4500)])
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D ,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy
import tensorflow.keras.backend as kb
#from solve_cudnn_error import *

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping()

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
import random
#data_path = '/home/allen/dl_grasp/src/data_expend/expand_data/5object_2500-11-11_07_30_55_.csv'

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, data_path, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.datas_path = data_path
        #self.datas = datas
        self.shuffle = shuffle
##########################################################
        data_frame = pd.read_csv(self.datas_path, sep=",")
        data1 = []
        data2 = []
        data3 = []
        data4 = []
   
        for indexe in data_frame.index:
            data_row = data_frame.loc[indexe]
            data1.append(data_row["image_path"])
            data2.append(data_row["target_x"])
            data3.append(data_row["target_y"])
            data4.append(data_row["target_angle"])
            ##############################
        for i in range(len(data2)):
            data2[i] = data2[i]-320
            data3[i] = data3[i]-240
            ##############################
        for i in range(len(data4)):
         #######################
            if data4[i]>90:
                data4[i] = -1*(data4[i]-90)
            elif data4[i]<=90:
                data4[i] = 90-data4[i]
        self.image_path = data1
        self.target_x = data2
        self.target_y = data3
        self.target_angle = data4
        self.indexes = np.arange(len(self.image_path))
###########################################################
    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(len(self.image_path) / float(self.batch_size))

    def __getitem__(self, index):
        #生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # 根据索引获取datas集合中的数据
        #batch_datas = [self.datas[k] for k in batch_indexs]
        # 生成数据
        X, y = self.data_generation(batch_indexs)

        return X, y

    def on_epoch_end(self):
        #在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas):
        

        # x_train
        photo_array= cv2.imread(self.image_path[batch_datas[0]],0)
        for i in range(len(batch_datas)-1):
            image=cv2.imread(self.image_path[batch_datas[i+1]],0)
            ###
            #image = image/255
            photo_array=np.concatenate((photo_array,image))
        photo_array=photo_array.reshape((-1,480,640,1))
        ########normalize
        photo_array=photo_array/255

        #y_train数据 
        result_array=np.zeros([len(batch_datas),4],  dtype=float)
        for i in range(len(batch_datas)):
            diameter = self.target_angle[batch_datas[i]]*math.pi/180
            cos_pi = math.cos(2*diameter)
            sin_pi = math.sin(2*diameter)
            result_array[i]=[self.target_x[batch_datas[i]]/320,self.target_y[batch_datas[i]]/240,cos_pi,sin_pi]
            #result_array[i]=[self.target_x[batch_datas[i]]/640,self.target_y[batch_datas[i]]/480,cos_pi,sin_pi]
            # print('cos : {} sin : {}'.format(cos_pi,sin_pi))
        return photo_array, result_array