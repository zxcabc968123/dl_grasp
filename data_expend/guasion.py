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
import random as rand
def gasuss_noise(image, mean, var):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    cv2.imshow('dsafsdfasdf',noise)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    #out = np.clip(out, low_clip, 1.0)
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out*255)
    cv2.imshow("gasuss", out)
    return out
def add_gaussian_noise(image_in, noise_sigma): 
    temp_image = np.float64(np.copy(image_in)) 
    h = temp_image.shape[0] 
    w = temp_image.shape[1] 
    noise = np.random.randn(h, w) * noise_sigma
    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2: 
        noisy_image = temp_image + noise 
    else: 
        noisy_image[:, :, 0] = temp_image[:, :, 0] + noise 


    print('min,max = ', np.min(noisy_image), np.max(noisy_image)) 
    print('type = ', type(noisy_image[0]))
    
    return noisy_image
if __name__ == "__main__":
    img_path = '/home/allen/dl_grasp/src/data_expend/origin_img/img/blackbox/blackbox_1.jpg' 

    img = cv2.imread(img_path,0) 
    cv2.imwrite('sourse.jpg',img)
    noise_sigma =  0
    noise_img = add_gaussian_noise(img, noise_sigma=noise_sigma) 
    noise_path = 'noise_{}.jpg'.format(noise_sigma) 
    cv2.imwrite(noise_path, noise_img) 
    cv2.imshow('source', img) 
    noise_img = cv2.imread(noise_path,0) 
    cv2.imshow('noise', noise_img)
    ###########################
    new_image=gasuss_noise(img,0,0.001)
    cv2.imshow('new_guasion',new_image)
    cv2.waitKey() 
    cv2.destroyAllWindows()



