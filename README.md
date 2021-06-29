# dcn-4-heads-yolov5
Add dcn and a 4 heads model config file in 'models' folder to process objects in different size
# Keypoint

1. how to use *github* to download code and find issues 
2. how to use code/folders downloaded ()
3. knowledge of python/pytorch/numpy etc. (how to *code* deeplearning model in python language)
4. how to use *colab/azure* to run code and setup environment for yolov5
5. knowlegde of deeplearning and YOLO 



# Github

- You can find the source code from [YOLOv5](https://github.com/ultralytics/yolov5)
- you can try to explore problems in *issue*s section
- use *git clone* to get the newest code 



# How to use ultralytics/YOLOv5 code?

## Ⅰ. Folders and files

### 1. data

#### **dataset.yaml**

- The yaml files in this folder aim to configure dataset used in training/testing/ process. You can change the dictionary in these files to make use of your own dataset.

- **Note:** directory of images and labels should be orgnized in example way, such as coco.yaml

#### class names in dataset.yaml

- items in this list should be in the sequence of label number from 0 in Label txt files 

#### hyp.yaml

- Hyperparameters setting
- scracth is used to training from the begining or to retrain a model
- finetune is used to training from pretrained model or to finetune models

### **2. models**

- You can configure yolov5 model here.

#### yolov5s/m/l/x.yaml

- Adjust model depth and with on the top of the file opened
- Models are modified in modular approch (**which modules are arranged in model**)

#### common.py

- Modules are coded here, such as C3, Conv, SPP etc. used in model's yaml (**what are modules**)

#### yolo.py

- Model construction based on common.py and yolov5.yaml is completed here (**how the models are construct in details**)
- When moudles added to common.py, yolo.py should also be modified somewhere concerned.

### 3. utils

- tool files 

#### general.py

#### loss.py

- to build targets of input, then compute loss
- loss functions are defined here

#### plots

- to plot figures of train/val results in results.txt
- to plot metrics

#### metrics

- to define how to compute precision/Recall/mAP ....

### 4. weights

- This folder contains weight files for yolov5 models, the files inside in .pt format and will be download automatically if not existed when you use scripts to train. The .pt files are general in deep learning codes, you may see .h5 or .pth weight files in other deep learning codes.

### 5.runs

- train/test/detect results saved here, figures contained

### 6.requirements.txt

- python packages list required

### 7.train.py  test.py  detect.py

- train.py: train a model and save in .pt file format
- test.py: to test a model trained on val/test dataset, and obtain the metrics/figures
- detect.py: use a trained model to detect pictures/video/camera data, and save results

# Good blogs: 

[usage of YOLOv5 code from download to train your own dataset](https://shliang.blog.csdn.net/article/details/106785253)

[analysis of train.py for YOLOv5](https://blog.csdn.net/Q1u1NG/article/details/107463417)

[analysis of test.py for YOLOv5](https://blog.csdn.net/Q1u1NG/article/details/107464724)

[how YOLOv5 algorithm work logically](https://blog.csdn.net/WZZ18191171661/article/details/113789486)



# What is YOLO?

YOLO models are one-stage object detection methods formulated by deep learning, whose name from "you only look once". You can learn yolo from this website: [zhihu-yolo series](https://zhuanlan.zhihu.com/p/183261974)



# Suggested learning steps of YOLOv5

1. knowledge about neural network (weights, linear and non-linear functions, optimizer, loss functions)
2. knowlege about object detection (mAP, one stage and two stage method)
3. knowlege about steps of construct a model in python and how to train (what's train/valitation/test)
