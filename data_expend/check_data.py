#!/usr/bin/env python3
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

data_csv='/home/allen/dl_grasp/src/data_expend/origin_data/data_2020-10-12.csv'

def pd_read_csv(csvFile):
 
    data_frame = pd.read_csv(csvFile, sep=",")
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    data6 = []
    data7 = []
    data8 = []

    for index in data_frame.index:
        data_row = data_frame.loc[index]
        data1.append(data_row["image_path"])
        data2.append(data_row["target_x"])
        data3.append(data_row["target_y"])
        data4.append(data_row["target_angle"])
        data5.append(data_row["ix"])
        data6.append(data_row["iy"])
        data7.append(data_row["width"])
        data8.append(data_row["high"])
    return (data1,data2,data3,data4,data5,data6,data7,data8)
    


def main():
    data1,data2,data3,data4,data5,data6,data7,data8=pd_read_csv(data_csv)
    for i in range(len(data1)-1):
        image=cv2.imread(data1[i],-1)
        cv2.circle(image, (data2[i],data3[i]), 5, (0,0,255),3)
        cv2.rectangle(image,(data5[i],data6[i]),(data7[i],data8[i])\
            ,(255, 0, 0),2)
        cv2.imshow('My image',image)
        cv2.waitKey(0)
if __name__ == "__main__":
    main()