from __future__ import absolute_import

import os
import glob
import numpy as np

##############################################################
import sys
current_dir = os.getcwd()  # obtain work dir
sys.path.append(current_dir)  # add work dir to sys path
##############################################################

from siamfc import TrackerSiamFC

## IF you want to get the results from the original SiameseFC, go to 'siamfc/__init__.py' (siamfc folder) and comment the
## the code ""


if __name__ == '__main__':

    ## Absolute PATH for data
    # seq_dir = os.path.expanduser('E:\PSUThirdSemester\CSE586ComputerVision\Term-Project1\Pythonversion\data\demo\Soccer')
    ## Relative PATH (if cannot not work using Absolute PATH based on your own computer)
    seq_dir = os.path.expanduser('..\..\.\data\demo\Soccer')
    # seq_dir = os.path.expanduser('..\..\.\data\demo\Basketball')

    ## differnt dataset you have to modify this line a little bit, for example you can check demo 2
    img_files = sorted(glob.glob(seq_dir + '/img/*.jpg'))
    anno = np.loadtxt(seq_dir + '/groundtruth_rect.txt')

    ## Absolute PATH for net
    # net_path = 'E:\PSUThirdSemester\CSE586ComputerVision\Term-Project1\Pythonversion\siamfc-pytorch-master\defaultpretrained\siamfc_alexnet_e1.pth'
    ## Relative PATH (if cannot not work using Absolute PATH based on your own computer)
    net_path = '.././defaultpretrained\siamfc_alexnet_e1.pth'

    tracker = TrackerSiamFC(net_path=net_path)

    tracker.track(img_files, anno[0], visualize=True)
