from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker

from . import ops
from .backbones import AlexNetV1
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms

##########################################
## 4 new import ##
from .KalmanFilter import KalmanFilter
from .DCF_net import DCFNet, CFConfig
import argparse
from .DCF_util import crop_chw,rect1_2_cxy_wh, cxy_wh_2_bbox


__all__ = ['TrackerSiamFC']


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)

        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()
        # print("loss function:")
        # print(self.criterion)

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)

        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

############################################################################################
        ## Initialization for Kalman Filter part
        # Create KalmanFilter object KF, and try different filter with different args
        # KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
        """
        :param dt: sampling time (time for 1 cycle)
        :param u_x: acceleration in x-direction
        :param u_y: acceleration in y-direction
        :param std_acc: process noise magnitude
        :param x_std_meas: standard deviation of the measurement in x-direction
        :param y_std_meas: standard deviation of the measurement in y-direction
        """

        #2-d constant speed - 2 constant accelerate
        # self.KF = KalmanFilter(0.2, 1.2, 1.2, 1, 0.1, 0.1)
        self.KF = KalmanFilter(0.2, 0.8, 0.8, 1, 0.1, 0.1)

############################################################################################
        ## some thresholds and parameters
        self.theta = 0.6
        self.beta1 = 0.25
        self.beta2 = 0.02
        self.kal_part = 0.75
############################################################################################


    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 32,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()
        box_1 = box
        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
                    self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)

        ######################################################################
        ## initialization for Correlation Filter part
        DCFparser = argparse.ArgumentParser(description='Test DCFNet on OTB')

        ## Absolute PATH for net
        # DCFparser.add_argument('--model', metavar='PATH', default= 'E:\PSUThirdSemester\CSE586ComputerVision\Term-Project1\Pythonversion\siamfc-pytorch-master\pretrained\DCF_param.pth')
        ## Relative PATH (if cannot not work using Absolute PATH based on your own computer)
        DCFparser.add_argument('--model', metavar='PATH', default= '..\.\defaultpretrained\CF_param.pth')

        DCFargs = DCFparser.parse_args()

        self.config_dcf = CFConfig()
        self.DCFnet = DCFNet(self.config_dcf).to(self.device)
        self.DCFnet.load_param(DCFargs.model)

        self.target_pos_dcf, self.target_sz_dcf = rect1_2_cxy_wh(box_1)
        window_sz_dcf = self.target_sz_dcf * (1 + self.config_dcf.padding)
        bbox_dcf = cxy_wh_2_bbox(self.target_pos_dcf, window_sz_dcf)
        patch_dcf = crop_chw(img, bbox_dcf, self.config_dcf.crop_sz)

        self.min_sz_dcf = np.maximum(self.config_dcf.min_scale_factor * self.target_sz_dcf, 4)
        self.max_sz_dcf = np.minimum(img.shape[:2], self.config_dcf.max_scale_factor * self.target_sz_dcf)

        target_dcf = patch_dcf - self.config_dcf.net_average_image
        self.DCFnet.update(torch.Tensor(np.expand_dims(target_dcf, axis=0)).to(self.device))

        self.patch_crop_dcf = np.zeros((self.config_dcf.num_scale, patch_dcf.shape[0], patch_dcf.shape[1], patch_dcf.shape[2]), np.float32)
        ######################################################################

    @torch.no_grad()
    def update(self, img, f):
        # set to evaluation mode
        self.net.eval()
        # print(self.center)
        # print([f,0])
        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()

        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
                   self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)
        # print(loc)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
                           self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
                        self.scale_factors[scale_id] / self.cfg.instance_sz

        # update target size
        scale = (1 - self.cfg.scale_lr) * 1.0 + \
                self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale
########################################################################################################################
        ## Modified the code for this part Combine with Kalman and Correlation Filters
########################################################################################################################
        centerFC = self.center + disp_in_image
        x_siamFC = centerFC[1] + 1 - (self.target_sz[1] - 1) / 2
        y_siamFC = centerFC[0] + 1 - (self.target_sz[0] - 1) / 2
        # Predict using kalman
        (x_kal, y_kal) = self.KF.predict()
        # Format changing
        X_array = np.array(x_kal)
        coordinate_X = X_array[0]
        x_kalman = coordinate_X[0]
        Y_array = np.array(y_kal)
        coordinate_Y = Y_array[0]
        y_kalman = coordinate_Y[0]
        # Compare the IoU for different method
        # box1 for FC
        box1 = np.array([
            x_siamFC,
            y_siamFC,
            self.target_sz[1], self.target_sz[0]])
        # box2 for FC with kalman
        box2 = np.array([
            x_kalman,
            y_kalman,
            self.target_sz[1], self.target_sz[0]])
        ious_FC_Kal = self.rect_iou(box1.T, box2.T)
        # print(ious)
        # theta is a threshold1
        # if f == 75:
        #     print("hhh")
        if (ious_FC_Kal >= self.theta) | (f <= 15):
            # Using Orignal FC
            x_final = x_siamFC
            y_final = y_siamFC
            self.CF_update_predict(img)
            method = 0
            # print("0")
        else:
            # Need Kalman motion model to modify the center get by Original FC
            x_kalman_new = self.kal_part*x_kalman+ (1 - self.kal_part)*x_siamFC
            y_kalman_new = self.kal_part*y_kalman+ (1 - self.kal_part)*y_siamFC
            # Need correlation Filter to justify the result and modify the result which is apparently wrong
            A, B = self.CF_update_predict(img)
            x_dcf = int(A[0] - B[0] / 2)
            y_dcf = int(A[1] - B[1] / 2)
            box_dcf = np.array([x_dcf, y_dcf, int(B[0]), int(B[1])])
            iou_dcf_fc = self.rect_iou(box1.T,box_dcf.T)
            iou_dcf_kal = self.rect_iou(box2.T,box_dcf.T)
            # print("#######################")
            # print(iou_dcf_fc)
            # print(iou_dcf_kal)
            if iou_dcf_fc > iou_dcf_kal:
                if iou_dcf_fc < self.beta1:
                    x_final = x_dcf
                    y_final = y_dcf
                    method = 1
                    # print("1")
                else:
                    x_final = x_siamFC
                    y_final = y_siamFC
                    method = 0
                    # print("2")
            else:
                if iou_dcf_kal > self.beta2:
                    x_final = x_dcf
                    y_final = y_dcf
                    method = 1
                    # print("3")
                else:
                    x_final = x_kalman_new
                    y_final = y_kalman_new
                    method = 2
                    # print("4")
    #############################################################################
        # return the box come back to self.center and update the data
        self.center[1] = x_final - 1 + (self.target_sz[1] - 1) / 2
        self.center[0] = y_final - 1 + (self.target_sz[0] - 1) / 2
        if method == 1:
            box_final = box_dcf
        else:
            box_final = np.array([
                x_final,
                y_final,
                self.target_sz[1], self.target_sz[0]])

        centers = np.array([[x_final], [y_final]])
        self.KF.update(centers)

        return box_final
#############################################################################################################
    ## Using for CF update predict
    def CF_update_predict(self, img):
        im = img  # img
        for i in range(self.config_dcf.num_scale):  # crop multi-scale search region
            window_sz_dcf = self.target_sz_dcf * (self.config_dcf.scale_factor[i] * (1 + self.config_dcf.padding))
            bbox_dcf = cxy_wh_2_bbox(self.target_pos_dcf, window_sz_dcf)
            self.patch_crop_dcf[i, :] = crop_chw(im, bbox_dcf, self.config_dcf.crop_sz)

        search_dcf = self.patch_crop_dcf - self.config_dcf.net_average_image
        response_dcf = self.DCFnet(torch.Tensor(search_dcf).to(self.device))
        peak_dcf, idx_dcf = torch.max(response_dcf.view(self.config_dcf.num_scale, -1), 1)
        idxcpu = idx_dcf.cpu()
        peakcpu = peak_dcf.data.cpu().numpy() * self.config_dcf.scale_penalties
        best_scale_dcf = np.argmax(peakcpu)
        r_max, c_max = np.unravel_index(idxcpu[best_scale_dcf], self.config_dcf.net_input_size)

        if r_max > self.config_dcf.net_input_size[0] / 2:
            r_max = r_max - self.config_dcf.net_input_size[0]
        if c_max > self.config_dcf.net_input_size[1] / 2:
            c_max = c_max - self.config_dcf.net_input_size[1]
        window_sz_dcf = self.target_sz_dcf * (self.config_dcf.scale_factor[best_scale_dcf] * (1 + self.config_dcf.padding))

        # print(np.array([c_max, r_max]) * window_sz / config.net_input_size)
        self.target_pos_dcf = self.target_pos_dcf + np.array([c_max, r_max]) * window_sz_dcf / self.config_dcf.net_input_size
        self.target_sz_dcf = np.minimum(np.maximum(window_sz_dcf / (1 + self.config_dcf.padding), self.min_sz_dcf), self.max_sz_dcf)

        window_sz_dcf = self.target_sz_dcf * (1 + self.config_dcf.padding)
        bbox_dcf = cxy_wh_2_bbox(self.target_pos_dcf, window_sz_dcf)
        patch_dcf = crop_chw(im, bbox_dcf, self.config_dcf.crop_sz)
        target_dcf = patch_dcf - self.config_dcf.net_average_image
        self.DCFnet.update(torch.Tensor(np.expand_dims(target_dcf, axis=0)).to(self.device), lr=self.config_dcf.interp_factor)

        # print(self.target_pos_dcf)
        # return target_pos_dcf, self.target_sz_dcf
        # return self.target_pos_dcf[0],self.target_pos_dcf[1],self.target_sz_dcf[0],self.target_sz_dcf[1]
        return self.target_pos_dcf, self.target_sz_dcf
####################################################################################################################

    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img, f)
            times[f] = time.time() - begin
            # print(boxes[f, :])
            if visualize:
                ops.show_image(img, boxes[f, :])

        return boxes, times
########################################################################################################################

    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)

            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='defaultpretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)

        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()

            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()

        return self.labels
########################################################################################################################
    def rect_iou(self, rects1, rects2, bound=None):
        r"""Intersection over union.

        Args:
            rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
                (left, top, width, height).
            rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
                (left, top, width, height).
            bound (numpy.ndarray): A 4 dimensional array, denotes the bound
                (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
        """
        assert rects1.shape == rects2.shape
        if bound is not None:
            # bounded rects1
            rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
            rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
            rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
            rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
            # bounded rects2
            rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
            rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
            rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
            rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

        x1 = np.maximum(rects1[..., 0], rects2[..., 0])
        y1 = np.maximum(rects1[..., 1], rects2[..., 1])
        x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                        rects2[..., 0] + rects2[..., 2])
        y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                        rects2[..., 1] + rects2[..., 3])

        w = np.maximum(x2 - x1, 0)
        h = np.maximum(y2 - y1, 0)

        rects_inter = np.stack([x1, y1, w, h]).T

        areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

        areas1 = np.prod(rects1[..., 2:], axis=-1)
        areas2 = np.prod(rects2[..., 2:], axis=-1)
        areas_union = areas1 + areas2 - areas_inter

        eps = np.finfo(float).eps
        ious = areas_inter / (areas_union + eps)
        ious = np.clip(ious, 0.0, 1.0)

        return ious