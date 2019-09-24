# ROS-based Grasp R-CNN for predict multi-grasp poses for multi-Objects 
This package is forked from https://github.com/ivalab/grasp_multiObject

## **Offical make steps [makebe not needed for this package, not sure]**

[Offical original version](https://github.com/ivalab/grasp_multiObject)

1. Build Cython modules
```
cd <ros_ws>/src/grasp_rcnn/scripts/grcnn/lib
make clean
make
cd ..
```

2. Install [Python COCO API](https://github.com/cocodataset/cocoapi)
```
cd <ros_ws>/src/grasp_rcnn/data
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
cd ../../..
```

## **Running grasp rcnn by ros (realsense d435i image as input)**

### Environment setting
- Remember to change python path in train.py and predict.py
- change this: sys.path.insert(1, "/home/<your_pc_name>/.local/lib/python3.5/site-packages/")
- python may be different for different pc, please check your path
- Download trained model for grasp on [dropbox drive](https://www.dropbox.com/s/ldapcpanzqdu7tc/models.zip?dl=0) 
- put under `<ros_ws>/src/grasp_rcnn/output/res50/train/default/`

### Running as a ros node
```  
cd src/rs_d435i/
source create_catkin_ws.sh
cd ../..
. /devel/setup.bash
roslaunch realsense2_camera rs_rgbd.launch
```

open new terminal    

```
. /devel/setup.bash 
. ../catkin_workspace/install/setup.bash --extend
cd <ros_ws>/src
rosrun grasp_rcnn demo_graspRGD.py
```

### Running by another node, grcnn as a python module
```  
cd src/rs_d435i/
source create_catkin_ws.sh
cd ../..
. /devel/setup.bash
roslaunch realsense2_camera rs_rgbd.launch
```

open new terminal  

```  
cd <ros_ws>/src
rosrun test_rs_img grcnn_module.py
```

## **Train with Cornell Grasp Dataset**
1. Generate data   
1-1. Download [Cornell Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php)   
1-2. Run `dataPreprocessingTest_fasterrcnn_split.m` (please modify paths according to your structure)   
1-3. Paste dataset to /home/<"user">/data_set/multi-grasp-dataset    (Dataset format: Follow 'Format Your Dataset' section [here](https://github.com/zeyuanxy/fast-rcnn/tree/master/help/train) to check if your data follows   VOC format)   
1-4. The dataset that augmentated based on Cornell Grasp Dataset is in [here](https://drive.google.com/drive/folders/1AVlwtJG-NlBSUeCGOy23H4a6jeMGodGm)    
**NOTE:** I completed step 1-1 and 1-2 in windows pc and completed other steps on ubuntu 

2. Train   
2-1. Download res50.ckpt model, which pre-trained on imagenet from [here](https://drive.google.com/drive/folders/1QHoi2JLj02UxOtt01ufwMJJEHhT632QN)   
2-2. Put it in `<ros_ws>/src/grasp_rcnn/data/imagenet_weights/res50.ckpt` 
2-3. cd <ros_ws>/src/grasp_rcnn    
2-4. Run `./experiments/scripts/train_faster_rcnn.sh 0 graspRGB res50`   

 
## **Acknowledgment and Reference**

This repo borrows tons of code from
- [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) by endernewton
- [grasp_multiObject_multiGrasp](https://github.com/ivalab/grasp_multiObject_multiGrasp) by ivalab

## **Resources**
- [multi-object grasp dataset](https://github.com/ivalab/grasp_multiObject)
- [grasp annotation tool](https://github.com/ivalab/grasp_annotation_tool)
