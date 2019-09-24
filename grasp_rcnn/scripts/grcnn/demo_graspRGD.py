#!/usr/bin/env python3

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import rospy
import grcnn._init_paths
from grcnn.lib.model.config import cfg
from grcnn.lib.model.test import im_detect
from grcnn.lib.model.nms_wrapper import nms

from grcnn.lib.utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import time
from grcnn.lib.nets.vgg16 import vgg16
from grcnn.lib.nets.resnet_v1 import resnetv1
import scipy
from shapely.geometry import Polygon

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from get_rs_image import Get_image

pi     = scipy.pi
dot    = scipy.dot
sin    = scipy.sin
cos    = scipy.cos
ar     = scipy.array

class Grasp_RCNN():
    def __init__(self):
        self.grasp_bboxes       = []
        self.grasp_angle_list   = []
        self.grasp_score_list   = []

        self.grasp_roi          = []
        self.grasp_scores       = []
        self.Enable_display     = True

        self.CLASSES = ('__background__',
                        'angle_01', 'angle_02', 'angle_03', 'angle_04', 'angle_05',
                        'angle_06', 'angle_07', 'angle_08', 'angle_09', 'angle_10',
                        'angle_11', 'angle_12', 'angle_13', 'angle_14', 'angle_15',
                        'angle_16', 'angle_17', 'angle_18', 'angle_19')

        self.NETS    = {'vgg16' : ('vgg16_faster_rcnn_iter_70000.ckpt',  ),
                        'res101': ('res101_faster_rcnn_iter_110000.ckpt',),
                        'res50' : ('res50_faster_rcnn_iter_240000.ckpt', )}

        self.DATASETS = {'pascal_voc'       : ('voc_2007_trainval',),
                         'pascal_voc_0712'  : ('voc_2007_trainval+voc_2012_trainval',),
                         'grasp'            : ('train',)}

        self.Enable_rpn_proposal = True
        cfg.TEST.HAS_RPN = self.Enable_rpn_proposal  # Use RPN for proposals
        # args = parse_args()
        
        # arg_set
        self.demonet = 'res50'
        self.dataset = 'grasp'
        self.ROOT_DIR = 'grasp_rcnn/output'
        self.tfmodel = os.path.join(self.ROOT_DIR, self.demonet, self.DATASETS[self.dataset][0], 'default',
                                    self.NETS[self.demonet][0])


        if not os.path.isfile(self.tfmodel + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                        'our server and place them properly?').format(self.tfmodel + '.meta'))

        # set config
        self.tfconfig = tf.ConfigProto(allow_soft_placement=True)
        self.tfconfig.gpu_options.allow_growth=True

        # init session
        self.sess = tf.Session(config = self.tfconfig)

        # load network
        if self.demonet == 'vgg16':
            self.net = vgg16(batch_size=1)

        elif self.demonet == 'res101':
            self.net = resnetv1(batch_size=1, num_layers=101)

        elif self.demonet == 'res50':
            self.net = resnetv1(batch_size=1, num_layers=50)

        else:
            raise NotImplementedError
        self.net.create_architecture(self.sess, "TEST", 20,
                            tag='default', anchor_scales=[8, 16, 32])

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.tfmodel)

        print('Loaded network {:s}'.format(self.tfmodel))


    def Rotate2D(self, pts,cnt,ang=scipy.pi/4):
        '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
        return dot(pts-cnt,ar([[cos(ang),sin(ang)],[-sin(ang),cos(ang)]]))+cnt

    def vis_detections(self, im, class_name, dets, cls_ind, thresh=0.5):
        """Draw detected bounding boxes (for each class)."""
        inds = np.where(dets[:, -1] >= thresh)[0]  # The roi index that score >= thresh score (0~1)
        if len(inds) == 0:
            return
        # print('inds = ', len(inds))
        # im = im[:, :, (2, 1, 0)]  # maybe get rgd data??
        
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            # plot rotated rectangles
            pts = ar([[bbox[0],bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
            cnt = ar([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
            angle = int(class_name[6:])
            r_bbox = self.Rotate2D(pts, cnt, -pi/2-pi/20*(angle-1))
            pred_label_polygon = Polygon([(r_bbox[0,0],r_bbox[0,1]), (r_bbox[1,0], r_bbox[1,1]), (r_bbox[2,0], r_bbox[2,1]), (r_bbox[3,0], r_bbox[3,1])])
            pred_x, pred_y = pred_label_polygon.exterior.xy

            # Output needed info
            self.grasp_bboxes.append( [ [int(pred_x[0]), int(pred_y[0])], 
                                        [int(pred_x[1]), int(pred_y[1])], 
                                        [int(pred_x[2]), int(pred_y[2])], 
                                        [int(pred_x[3]), int(pred_y[3])]  ] )
            self.grasp_angle_list.append(cls_ind*5)
            self.grasp_score_list.append(score)

        if(self.Enable_display==True):
            cv2.line(im, (int(pred_x[0]), int(pred_y[0])), (int(pred_x[1]), int(pred_y[1])), (0,0,  0), 1)
            cv2.line(im, (int(pred_x[1]), int(pred_y[1])), (int(pred_x[2]), int(pred_y[2])), (0,0,255), 2)
            cv2.line(im, (int(pred_x[2]), int(pred_y[2])), (int(pred_x[3]), int(pred_y[3])), (0,0,  0), 1)
            cv2.line(im, (int(pred_x[3]), int(pred_y[3])), (int(pred_x[0]), int(pred_y[0])), (0,0,255), 2)
            cv2.imshow('grasp detection', im)
            choice = cv2.waitKey(1)

    def demo(self, sess, net, image):
        """Detect object classes in an image using pre-computed object proposals."""

        # Load the demo image
        # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
        # im = cv2.imread(im_file)
        im = image

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        start = time.time()
        self.grasp_scores, self.grasp_roi = im_detect(sess, net, im)

        end = time.time()

        fps = 1 / (end-start)
        # print('=============')
        # print('Grasp R-CNN FPS = ', round(fps, 2))
        # print('=============')

        timer.toc()
        print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, self.grasp_roi.shape[0]))

        # Visualize detections for each class
        CONF_THRESH = 0.1	   
        NMS_THRESH  = 0.3

        # Reset needed grasp info
        self.grasp_score_list = []
        self.grasp_bboxes     = []      
        self.grasp_angle_list = []

        for cls_ind, cls in enumerate(self.CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = self.grasp_roi[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = self.grasp_scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            self.vis_detections(im, cls, dets, cls_ind, thresh=CONF_THRESH)
        

    def parse_args(self):
        """Parse input arguments."""
        parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
        parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                            choices=NETS.keys(), default='res101')
        parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                            choices=DATASETS.keys(), default='pascal_voc_0712')
        args = parser.parse_args()

        return args

if __name__ == '__main__':
    rospy.init_node('grasp_rcnn_predict', anonymous=True)
    sub_img = Get_image()
    grcnn = Grasp_RCNN()
    
    while not rospy.is_shutdown():
        img = np.asarray(sub_img.cv_image)
        if(img.size > 1):
            grcnn.demo(grcnn.sess, grcnn.net, img)

    rospy.spin()