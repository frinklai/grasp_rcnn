# ROS-based Grasp R-CNN for predict multi-grasp poses for multi-Objects 
This package is forked from https://github.com/ivalab/grasp_multiObject

## **Running grasp rcnn by ros (realsense d435i image as input)**

### Environment setting
- Remember to change python path in train.py and predict.py
- change this: sys.path.insert(1, "/home/<your_pc_name>/.local/lib/python3.5/site-packages/")
- python may be different for different pc, please check your path
- Download trained model for grasp on [dropbox drive](https://www.dropbox.com/s/ldapcpanzqdu7tc/models.zip?dl=0) 
- put under `<ros_ws>/src/<repo>/grasp_rcnn/output/res50/train/default/`

### Running as a ros node
```  
cd <ros_ws>/src/<repo>/
source create_catkin_ws.sh
cd <ros_ws>
. /devel/setup.bash
roslaunch realsense2_camera rs_rgbd.launch
```

open new terminal    

```
. /devel/setup.bash 
. ../catkin_workspace/install/setup.bash --extend
cd <ros_ws>/src/<repo>/
rosrun grasp_rcnn demo_graspRGD.py
```

### Running by another node, grcnn as a python module
```  
cd <ros_ws>/src/<repo>/
source create_catkin_ws.sh
cd <ros_ws>
. /devel/setup.bash
roslaunch realsense2_camera rs_rgbd.launch
```

open new terminal   

```  
cd <ros_ws>/src/<repo>
rosrun custom_stra grcnn_module.py
```

## **Acknowledgment and Reference**

This repo borrows tons of code from
- [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) by endernewton
- [grasp_multiObject_multiGrasp](https://github.com/ivalab/grasp_multiObject_multiGrasp) by ivalab
