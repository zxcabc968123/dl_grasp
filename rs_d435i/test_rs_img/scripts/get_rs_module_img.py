#!/usr/bin/env python3
import sys
sys.path.insert(1, "/home/allen/.local/lib/python3.5/site-packages/")
sys.path.insert(2, "/home/allen/realsensepkg/catkin_workspace/install/lib/python3/dist-packages")
import rospy
import cv2
from get_rs_image import Get_image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from pynput import keyboard
import time
path_depth = '/home/allen/dl_grasp/src/rs_d435i/pic/depth/'
pic_num=179
if __name__ == '__main__':
    rospy.init_node('get_d435i_module_image', anonymous=True)
    listener = Get_image()
    
    while not rospy.is_shutdown():
        listener.display_mode = 'depth'
        if(listener.display_mode == 'rgb')and(type(listener.cv_image) is np.ndarray):
            cv2.imshow("rgb module image", listener.cv_image)
        elif(listener.display_mode == 'depth')and(type(listener.cv_depth) is np.ndarray):

            abc=cv2.cvtColor(listener.cv_depth,cv2.COLOR_BGR2GRAY)
            cv2.imshow("depth module image", abc)

            key_num = cv2.waitKey(10)
            #print(key_num)
            if key_num == 115:
                cv2.imwrite(path_depth+str(pic_num) + '.jpg',listener.cv_depth)
                print('Save picture '+str(pic_num))
                pic_num=pic_num+1
        else:
            pass
        cv2.waitKey(1)

    rospy.spin()

