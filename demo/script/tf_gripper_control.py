#!/usr/bin/env python
# encoding: utf-8   #要打中文時加這行
import rospy
import sys
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output as outputmsg

class robotiq_2f_control:
    def __init__(self):
        self.topic_name = '/Robotiq2FGripperRobotOutput'
        self.publisher_rate=50
        self.pub_cmd=rospy.Publisher(self.topic_name,outputmsg,queue_size=10)
        self.speed = 255
        self.force = 150

    def gripper_control_reset(self):
        cmd = outputmsg()
        cmd.rACT = 0
        cmd.rGTO = 0
        cmd.rATR = 0 
        cmd.rPR  = 0
        cmd.rSP  = 0
        cmd.rFR  = 0
        self.pub_cmd.publish(cmd)
        rate=rospy.Rate(self.publisher_rate)

    def gripper_control_open(self):
        cmd = outputmsg()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rATR = 0 
        cmd.rPR  = 0
        cmd.rSP  = int(self.speed)
        cmd.rFR  = int(self.force)
        self.pub_cmd.publish(cmd)
        rate=rospy.Rate(self.publisher_rate)

    def gripper_control_close(self):
        cmd = outputmsg()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rATR = 0 
        cmd.rPR  = 255
        cmd.rSP  = int(self.speed)
        cmd.rFR  = int(self.force)
        self.pub_cmd.publish(cmd)
        rate=rospy.Rate(self.publisher_rate)
if __name__ == "__main__":
    rospy.init_node('gripper_control',anonymous=False)
    a=robotiq_2f_control()
    a.gripper_control_reset()
    rospy.sleep(1)
    a.gripper_control_close()
    rospy.sleep(1)
    a.gripper_control_open()