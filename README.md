# SiamFC-KF-CF

## Ke LIANG, Lin LI, Yifei XIANG, Lindsey Schwartz 

## Installation

Set up the environment

```bash
# install Anaconda
https://www.anaconda.com/distribution/
# install PyTorch >= 1.0
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
# intall OpenCV using menpo channel (otherwise the read data could be inaccurate)
conda install -c menpo opencv
# install GOT-10k toolkit
pip install got10k
# install git
different system can use different ways
# install got-10k library
pip install --upgrade got10k
pip install --upgrade git+https://github.com/got-10k/toolkit.git@master
```

## Training the tracker

1. Setup the training dataset in `test_and_evaluation/train.py`. Default is the GOT-10k dataset located at `Dir of your data for GOT-10k`.

2. execute test_and_evaluation/train.py

Notice the path is important to make things goes well, follow the comments


## Evaluate the tracker

1. Setup the tracking dataset in `test_and_evaluation/dEvaluation_test/test.py`. We offer 7 test code for 7 datasets,
follow the comments in the code.

2. Setup the checkpoint path of your pretrained models. 
Default is `test_and_evaluation/defaultpretrained/siamfc_alexnet_e1.pth` for original siameseFC.
Default is `test_and_evaluation/defaultpretrained/CF_param.pth` for original CF_net.


3. execute test_and_evaluation/train.py lktest_xxxxx.py

Notice the path is important to make things goes well, follow the comments

## Running the demo

1. Check the data for demos. Default path is in `data/demo`, and there are 3 demos.

2. Setup the checkpoint path of your pretrained model. 
Default is `test_and_evaluation/defaultpretrained/siamfc_alexnet_e1.pth` for original siameseFC.
Default is `test_and_evaluation/defaultpretrained/CF_param.pth` for original CF_net.

3. execute the demo1.py and demo2.py, follow the comments in the demo files.

Notice the path is important to make things goes well, follow the comments

## Module

All the tracker files are in the siamfc folder.

Besides modify the original codes in the beginning to make the original code run, we create 5 new files, as shown below.

KalmanFilter.py 
CF_net.py 
CF_util.py 
siamfc_Kalman_Correlation.py
siamfc_orignialFC.py (this is the modified version of the original code)

## Reference
We read and understand the their existed work, and create our term-proj

https://github.com/huanglianghua/siamfc-pytorch

https://github.com/got-10k/toolkit

https://github.com/RahmadSadli/2-D-Kalman-Filter

https://github.com/foolwood/DCFNet_pytorch

Luka Bertinetto, Jack Valmadre.Fully Convolutional Siamese Networks for Object Tracking.2015

Shiuh-Ku Weng, Chung-Ming Kuo, Shu-Kang Tu.Video object tracking using adaptive Kalman filter.2006

Vishnu Naresh Boddeti, Vijayakumar Bhagavatula.Advances in correlation filters: vector features, structured prediction and shape alignment.2011

David S. Bolme, J. Ross Beveridge Bruce, A. Draper Yui Man Lui.Visual Object Tracking using Adaptive Correlation Filters.2010

Chenlong Wu, Yue Zhang, Yi Zhang.Motion guided Siamese trackers for visual tracking.2020

Lijun Zhou, Jianlin Zhang.Combined Kalman Filter and Multifeature Fusion Siamese Network for Real-Time Visual Tracking.2019

Wang, Qiang and Gao, Jin and Xing, Junliang and Zhang, Mengdan and Hu, Weiming.DCFNet: Discriminant Correlation Filters Network for Visual Tracking.2017
