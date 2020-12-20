#!/usr/bin/env python
# encoding: utf-8   #要打中文時加這行
import rospy
import sys
from demo.srv import lungrasp 


def main():
    rospy.init_node('lun_client', anonymous=True) #初始化node 可不用 看下面
    rospy.loginfo('--------------------------------')
    rospy.wait_for_service('grasp_detection',timeout=1)     #等待server端開啟 service_p service名稱 timeout=等待時間 可為None
    x = True
    try:
        print('sdafsdafsdafdsfafsaf')
        client=rospy.ServiceProxy('grasp_detection',lungrasp) #定義client打給service 的函式
        resp1=client(x)   #打給service  並將收到的 response 存到 resp1   x,y,z 按照 檔的 順序
        print("confi : %r",resp1.confi)
        print("center_point_x : %r",resp1.center_point_x)
        print("center_point_y : %r",resp1.center_point_y)
        print("center_point_z : %r",resp1.center_point_z)
        print("ang : %r",resp1.ang)
        print("is_done : %r",resp1.is_done)
        print('sdafsdafsdafdsfafsaf')
        #print('%d  %s'%(resp1.ans,resp1.data))
    except rospy.ServiceException as exc:
        rospy.loginfo('end')
if __name__ == "__main__":
    main()