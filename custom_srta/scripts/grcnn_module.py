#!/usr/bin/env python3

import sys
sys.path.insert(1, "/home/iarc/.local/lib/python3.5/site-packages/")
import rospy
import cv2
from get_rs_image import Get_image
from grcnn import Grasp_RCNN
from cv_bridge import CvBridge, CvBridgeError
import numpy as np



if __name__ == '__main__':
    rospy.init_node('get_d435i_module_image', anonymous=True)
    sub_img = Get_image()
    grcnn1 = Grasp_RCNN()
    
    while not rospy.is_shutdown():
        img = np.asarray(sub_img.cv_image)
        if(img.size > 1):
            grcnn1.demo(grcnn1.sess, grcnn1.net, img)

    rospy.spin()

