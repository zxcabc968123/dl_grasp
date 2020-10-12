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
def main():
    test_image_path='/home/allen/dl_grasp/src/rs_d435i/pic/depth_train/locate/6.jpg'
    image=cv2.imread(test_image_path,0)
    cv2.imshow('My Image', image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()