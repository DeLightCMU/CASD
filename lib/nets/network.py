# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.visualization import draw_bounding_boxes

from ops.roi_pool import RoIPool
from ops.roi_align import RoIAlign
from ops.roi_ring_pool import RoIRingPool

from model.config import cfg
from utils.bbox import bbox_overlaps
import tensorboardX as tb

from scipy.misc import imresize
from sklearn.cluster import KMeans
import random
from torch.autograd import Variable

import numpy.random as npr

import math

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self._predictions = {}
        self._losses = {}
        self._layers = {}
        self._gt_image = None
        self._event_summaries = {}
        self._image_gt_summaries = {}
        self._device = 'cuda'
        self.RoIPool = RoIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1. / 16)
        self.RoIAlign = RoIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1. / 16)
        self.RoIRingPool = RoIRingPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1. / 16, 0., 1.0)
        self.RoIRingPool_context = RoIRingPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1. / 16, 1.0, 1.8)
        self.RoIRingPool_frame = RoIRingPool(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1. / 16, 1.0 / 1.8, 1.0)
        self.aug_time = 4
        self.ca_iw = True

    def _add_gt_image(self):
        # add back mean
        image = self._image_gt_summaries['image'] + cfg.PIXEL_MEANS
        image = imresize(image[0], self._im_info[:2] / self._im_info[2])
        # BGR to RGB (opencv uses BGR)
        self._gt_image = image[np.newaxis, :, :, ::-1].copy(order='C')

    def _add_gt_image_summary(self):
        # use a customized visualization function to visualize the boxes
        self._add_gt_image()
        image = draw_bounding_boxes(self._gt_image, self._image_gt_summaries['gt_boxes'], self._image_gt_summaries['im_info'])
        return tb.summary.image('GROUND_TRUTH', image[0].astype('float32').swapaxes(1, 0).swapaxes(2, 0) / 255.0)

    def _inverted_attention(self, bbox_feats_new, gt, keep_inds, refine_branch, step, fg_num, bg_num):
        if step <= (cfg.TRAIN.STEP_ITERS + 10000):
            fg_drop_per = 6
            bg_drop_per = fg_drop_per * 3
            th_l = 25
            th_s = 16
        else:
            fg_drop_per = 5
            bg_drop_per = fg_drop_per * 3
            th_l = 29
            th_s = 20

        self.eval()
        bbox_feats_new = Variable(bbox_feats_new.data, requires_grad=True)
        bbox_feats_new_new = self._head_to_tail(bbox_feats_new)
        if refine_branch == 1:
            output_score = self.refine_net_1(bbox_feats_new_new)
        elif refine_branch == 2:
            output_score = self.refine_net_2(bbox_feats_new_new)
        else:
            print('no refine branch')
        class_num = output_score.shape[1]
        index = gt
        num_rois = bbox_feats_new.shape[0]
        num_channel = bbox_feats_new.shape[1]
        one_hot = torch.zeros((1), dtype=torch.float32).cuda()
        one_hot = Variable(one_hot, requires_grad=False)
        sp_i = torch.ones([2, num_rois]).long()
        sp_i[0, :] = torch.arange(num_rois)
        sp_i[1, :] = torch.from_numpy(index)
        sp_v = torch.ones([num_rois])
        one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
        one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)
        one_hot = torch.sum(output_score * one_hot_sparse)
        self.vgg.classifier.zero_grad()
        if refine_branch == 1:
            self.refine_net_1.zero_grad()
        elif refine_branch == 2:
            self.refine_net_2.zero_grad()
        else:
            print('no refine branch')
        one_hot.backward()
        grads_val = bbox_feats_new.grad.clone().detach()
        grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
        grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
        att_all = torch.sum(bbox_feats_new * grad_channel_mean, 1)
        att_all = att_all.view(num_rois, 49)

        self.vgg.classifier.zero_grad()
        if refine_branch == 1:
            self.refine_net_1.zero_grad()
        elif refine_branch == 2:
            self.refine_net_2.zero_grad()
        else:
            print('no refine branch')

        thl_mask_value = torch.sort(att_all, dim=1, descending=True)[0][:, th_l]
        thl_mask_value = thl_mask_value.view(num_rois, 1).expand(num_rois, 49)
        mask_all_cuda = torch.where(att_all > thl_mask_value, torch.zeros(att_all.shape).cuda(), torch.ones(att_all.shape).cuda())
        mask_all = mask_all_cuda.detach().cpu().numpy()
        mask_all_new = np.ones((num_rois, 49), dtype=np.float32)
        for q in keep_inds:
            mask_all_temp = np.ones((49), dtype=np.float32)
            zero_index = np.where(mask_all[q, :] == 0)[0]
            num_zero_index = zero_index.size
            if num_zero_index >= th_s:
                dumy_index = npr.choice(zero_index, size=th_s, replace=False)
            else:
                zero_index = np.arange(49)
                dumy_index = npr.choice(zero_index, size=th_s, replace=False)
            mask_all_temp[dumy_index] = 0
            mask_all_new[q, :] = mask_all_temp
        mask_all = torch.from_numpy(mask_all_new.reshape(num_rois, 7, 7)).cuda()
        mask_all = mask_all.view(num_rois, 1, 7, 7)

        pooled_feat_before_after = torch.cat((bbox_feats_new, bbox_feats_new * mask_all), dim=0)
        pooled_feat_before_after = self._head_to_tail(pooled_feat_before_after)
        if refine_branch == 1:
            cls_score_before_after = self.refine_net_1(pooled_feat_before_after)
        elif refine_branch == 2:
            cls_score_before_after = self.refine_net_2(pooled_feat_before_after)
        else:
            print('no refine branch')
        cls_prob_before_after = F.softmax(cls_score_before_after, dim=1)
        class_num = cls_prob_before_after.shape[1]
        cls_prob_before = cls_prob_before_after[0: num_rois]
        cls_prob_after = cls_prob_before_after[num_rois: num_rois * 2]
        label_gt = torch.from_numpy(gt).cuda()
        prepare_mask_fg_num = fg_num
        prepare_mask_bg_num = bg_num
        sp_i = torch.ones([2, num_rois]).long()
        sp_i[0, :] = torch.arange(num_rois)
        sp_i[1, :] = label_gt
        sp_v = torch.ones([num_rois])
        one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
        before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
        after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
        change_vector = before_vector - after_vector - 0.02
        change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
        fg_index = torch.where(label_gt > 0, torch.ones(before_vector.shape).cuda(), torch.zeros(before_vector.shape).cuda())
        bg_index = 1 - fg_index

        if fg_index.nonzero().shape[0] != 0:
            not_01_fg_index = fg_index.nonzero()[:, 0].long()
        else:
            not_01_fg_index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).cuda().long()  # for corner case
        not_01_bg_index = bg_index.nonzero()[:, 0].long()
        change_vector_fg = change_vector[not_01_fg_index]
        change_vector_bg = change_vector[not_01_bg_index]
        for_fg_change_vector = change_vector.clone()
        for_bg_change_vector = change_vector.clone()
        for_fg_change_vector[not_01_bg_index] = -10000
        for_bg_change_vector[not_01_fg_index] = -10000

        th_fg_value = torch.sort(change_vector_fg, dim=0, descending=True)[0][int(round(float(prepare_mask_fg_num) / fg_drop_per))]
        drop_index_fg = for_fg_change_vector.gt(th_fg_value)
        th_bg_value = torch.sort(change_vector_bg, dim=0, descending=True)[0][int(round(float(prepare_mask_bg_num) / bg_drop_per))]
        drop_index_bg = for_bg_change_vector.gt(th_bg_value)
        drop_index_fg_bg = drop_index_fg + drop_index_bg
        ignore_index_fg_bg = 1 - drop_index_fg_bg
        not_01_ignore_index_fg_bg = ignore_index_fg_bg.nonzero()[:, 0]
        mask_all[not_01_ignore_index_fg_bg.long(), :] = 1
        self.train()
        return mask_all

    def _normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins + 1e-7)
        atten_normed = atten_normed.view(atten_shape)
        return atten_normed

    def _rampweight(self, iteration):
        ramp_up_end = 45000
        ramp_down_start = 100000
        if (iteration < ramp_up_end):
            ramp_weight = math.exp(-5 * math.pow((1 - iteration / ramp_up_end), 2))
        elif (iteration > ramp_down_start):
            ramp_weight = math.exp(-12.5 * math.pow((1 - (100000 - iteration) / 20000), 2))
        else:
            ramp_weight = 1

        if (iteration <= 45000):
            ramp_weight = 0
            self.ca_iw = False
        return ramp_weight

    def _add_losses(self, roi_labels_1, keep_inds_1, roi_labels_2, keep_inds_2, step=None, rois=None):
        det_cls_prob = self._predictions['det_cls_prob']
        det_cls_prob = det_cls_prob.view(-1)
        label = self._image_level_label.view(-1)
        pi = self.ss_boxes_indexes.shape[0]
        rampweight = self._rampweight(step)

        refine_prob_1 = self._predictions['refine_prob_1']
        refine_prob_2 = self._predictions['refine_prob_2']
        # refine_prob_3 = self._predictions['refine_prob_3']

        # caculating the loss of the first branch
        roi_labels, keep_inds = roi_labels_1, keep_inds_1,
        roi_labels_each = torch.tensor(roi_labels[0][keep_inds[0], :], dtype=torch.float32).cuda()
        refine_loss_1 = - torch.sum(torch.mul(roi_labels_each, torch.log(refine_prob_1[keep_inds[0]]))) / roi_labels_each.shape[0]
        roi_labels_each = torch.tensor(roi_labels[1][keep_inds[1], :], dtype=torch.float32).cuda()
        refine_loss_1 -= torch.sum(torch.mul(roi_labels_each, torch.log(refine_prob_1[keep_inds[1] + pi]))) / roi_labels_each.shape[0]
        roi_labels_each = torch.tensor(roi_labels[2][keep_inds[2], :], dtype=torch.float32).cuda()
        refine_loss_1 -= torch.sum(torch.mul(roi_labels_each, torch.log(refine_prob_1[keep_inds[2] + pi * 2]))) / roi_labels_each.shape[0]
        roi_labels_each = torch.tensor(roi_labels[3][keep_inds[3], :], dtype=torch.float32).cuda()
        refine_loss_1 -= torch.sum(torch.mul(roi_labels_each, torch.log(refine_prob_1[keep_inds[3] + pi * 3]))) / roi_labels_each.shape[0]

        consistency_conf_loss = 0
        if self.ca_iw:
            keep_inds_new = np.concatenate((keep_inds[0], keep_inds[0]+pi, keep_inds[0]+pi*2, keep_inds[0]+pi*3))
            num_each = int(keep_inds_new.shape[0] / 4)
            rois_self_attention_1 = torch.mean(rois[keep_inds_new], dim=1)
            rois_self_attention_1 = torch.sigmoid(self._normalize_atten_maps(rois_self_attention_1))

            rois_self_attention_gt_1 = rois_self_attention_1.clone().detach()
            rois_self_attention_gt_1_1 = torch.max(rois_self_attention_gt_1[0:num_each], rois_self_attention_gt_1[num_each:num_each * 2].flip(dims=[2]))
            rois_self_attention_gt_1_2 = torch.max(rois_self_attention_gt_1[num_each * 2:num_each * 3], rois_self_attention_gt_1[num_each * 3:num_each * 4].flip(dims=[2]))
            rois_self_attention_gt_1 = torch.max(rois_self_attention_gt_1_1, rois_self_attention_gt_1_2)
            consistency_conf_loss += F.mse_loss(rois_self_attention_1[0:num_each], rois_self_attention_gt_1)
            consistency_conf_loss += F.mse_loss(rois_self_attention_1[num_each:num_each * 2], rois_self_attention_gt_1.flip(dims=[2]))
            consistency_conf_loss += F.mse_loss(rois_self_attention_1[num_each * 2:num_each * 3], rois_self_attention_gt_1)
            consistency_conf_loss += F.mse_loss(rois_self_attention_1[num_each * 3:num_each * 4], rois_self_attention_gt_1.flip(dims=[2]))

        # caculating the loss of the second branch
        roi_labels, keep_inds = roi_labels_2, keep_inds_2
        roi_labels_each = torch.tensor(roi_labels[0][keep_inds[0], :], dtype=torch.float32).cuda()
        refine_loss_2 = - torch.sum(torch.mul(roi_labels_each, torch.log(refine_prob_2[keep_inds[0]]))) / roi_labels_each.shape[0]
        roi_labels_each = torch.tensor(roi_labels[1][keep_inds[1], :], dtype=torch.float32).cuda()
        refine_loss_2 -= torch.sum(torch.mul(roi_labels_each, torch.log(refine_prob_2[keep_inds[1] + pi]))) / roi_labels_each.shape[0]
        roi_labels_each = torch.tensor(roi_labels[2][keep_inds[2], :], dtype=torch.float32).cuda()
        refine_loss_2 -= torch.sum(torch.mul(roi_labels_each, torch.log(refine_prob_2[keep_inds[2] + pi * 2]))) / roi_labels_each.shape[0]
        roi_labels_each = torch.tensor(roi_labels[3][keep_inds[3], :], dtype=torch.float32).cuda()
        refine_loss_2 -= torch.sum(torch.mul(roi_labels_each, torch.log(refine_prob_2[keep_inds[3] + pi * 3]))) / roi_labels_each.shape[0]

        if self.ca_iw:
            keep_inds_new = np.concatenate((keep_inds[0], keep_inds[0]+pi, keep_inds[0]+pi*2, keep_inds[0]+pi*3))
            num_each = int(keep_inds_new.shape[0] / 4)
            rois_self_attention_2 = torch.mean(rois[keep_inds_new], dim=1)
            rois_self_attention_2 = torch.sigmoid(self._normalize_atten_maps(rois_self_attention_2))

            rois_self_attention_gt_2 = rois_self_attention_2.clone().detach()
            rois_self_attention_gt_2_1 = torch.max(rois_self_attention_gt_2[0:num_each], rois_self_attention_gt_2[num_each:num_each*2].flip(dims=[2]))
            rois_self_attention_gt_2_2 = torch.max(rois_self_attention_gt_2[num_each*2:num_each*3], rois_self_attention_gt_2[num_each*3:num_each*4].flip(dims=[2]))
            rois_self_attention_gt_2 = torch.max(rois_self_attention_gt_2_1, rois_self_attention_gt_2_2)
            consistency_conf_loss += F.mse_loss(rois_self_attention_2[0:num_each], rois_self_attention_gt_2)
            consistency_conf_loss += F.mse_loss(rois_self_attention_2[num_each:num_each*2], rois_self_attention_gt_2.flip(dims=[2]))
            consistency_conf_loss += F.mse_loss(rois_self_attention_2[num_each*2:num_each*3], rois_self_attention_gt_2)
            consistency_conf_loss += F.mse_loss(rois_self_attention_2[num_each*3:num_each*4], rois_self_attention_gt_2.flip(dims=[2]))
            consistency_conf_loss /= num_each

        label_new = torch.cat((label, label, label, label))
        label_new = label_new.clone().detach().float().to(det_cls_prob.device)
        zeros = torch.zeros(det_cls_prob.shape, dtype=det_cls_prob.dtype, device=det_cls_prob.device)
        max_zeros = torch.max(zeros, 1 - torch.mul(label_new, det_cls_prob))
        cls_det_loss = torch.sum(max_zeros)
        loss = cls_det_loss / 20 + refine_loss_1 * 0.1 + refine_loss_2 * 0.1 + consistency_conf_loss * 0.1
        loss /= float(self.aug_time)

        self._losses['total_loss'] = loss
        self._losses['cls_det_loss'] = cls_det_loss / 20
        self._losses['refine_loss_1'] = refine_loss_1 * 0.1
        self._losses['refine_loss_2'] = refine_loss_2 * 0.1
        # self._losses['refine_loss_3'] = refine_loss_3
        if self.ca_iw is False:
            consistency_conf_loss = torch.zeros([1])
        self._losses['consistency_loss'] = consistency_conf_loss
        for k in self._losses.keys():
            self._event_summaries[k] = self._losses[k]
        return loss

    def _region_classification_test(self, fc7_roi, fc7_context, fc7_frame):
        refine_score_1 = self.refine_net_1(fc7_roi)
        refine_score_2 = self.refine_net_2(fc7_roi)
        # refine_score_3 = self.refine_net_3(fc7_roi)
        cls_score = self.cls_score_net(fc7_roi)
        context_score = self.det_score_net(fc7_context)
        frame_score = self.det_score_net(fc7_frame)
        det_score = frame_score - context_score

        cls_prob = F.softmax(cls_score, dim=1)
        det_prob = F.softmax(det_score, dim=0)
        refine_prob_1 = F.softmax(refine_score_1, dim=1)
        refine_prob_2 = F.softmax(refine_score_2, dim=1)
        # refine_prob_3 = F.softmax(refine_score_3, dim=1)

        det_cls_prob_product = torch.mul(cls_score, det_prob)
        det_cls_prob = torch.sum(det_cls_prob_product, 0)
        # bbox_pred = self.bbox_pred_net(fc7)
        bbox_pred = torch.zeros(cls_prob.shape[0], 80)

        self._predictions['refine_prob_1'] = refine_prob_1
        self._predictions['refine_prob_2'] = refine_prob_2
        # self._predictions['refine_prob_3'] = refine_prob_3
        self._predictions["bbox_pred"] = bbox_pred
        self._predictions['det_cls_prob_product'] = det_cls_prob_product
        self._predictions['det_cls_prob'] = det_cls_prob
        return cls_prob, det_prob, bbox_pred, det_cls_prob_product, det_cls_prob

    def _region_classification_train(self, pool5_roi, fc7_roi, fc7_context, fc7_frame, step):
        bbox_feats_new = pool5_roi.clone().detach()
        fc7_roi_new = fc7_roi.clone().detach()
        refine_score_1_new = self.refine_net_1(fc7_roi_new)
        refine_prob_1_new = F.softmax(refine_score_1_new, dim=1)

        cls_score = self.cls_score_net(fc7_roi)
        context_score = self.det_score_net(fc7_context)
        frame_score = self.det_score_net(fc7_frame)
        det_score = frame_score - context_score

        cls_prob = F.softmax(cls_score, dim=1)
        pi = self.ss_boxes_indexes.shape[0]
        if self._mode == 'TRAIN':
            ss_rois_num_each = int(cls_score.shape[0] / 4)
            assert ss_rois_num_each == pi
            det_prob1 = F.softmax(det_score[ss_rois_num_each * 0: ss_rois_num_each * 1], dim=0)
            det_prob2 = F.softmax(det_score[ss_rois_num_each * 1: ss_rois_num_each * 2], dim=0)
            det_prob3 = F.softmax(det_score[ss_rois_num_each * 2: ss_rois_num_each * 3], dim=0)
            det_prob4 = F.softmax(det_score[ss_rois_num_each * 3: ss_rois_num_each * 4], dim=0)
            det_prob = torch.cat((det_prob1, det_prob2, det_prob3, det_prob4))
        else:
            det_prob = F.softmax(det_score, dim=0)

        det_cls_prob_product = torch.mul(cls_score, det_prob)
        if self._mode == 'TRAIN':
            ss_rois_num_each = int(det_cls_prob_product.shape[0] / 4)
            assert ss_rois_num_each == pi
            det_cls_prob1 = torch.sum(det_cls_prob_product[ss_rois_num_each * 0: ss_rois_num_each * 1], 0)
            det_cls_prob2 = torch.sum(det_cls_prob_product[ss_rois_num_each * 1: ss_rois_num_each * 2], 0)
            det_cls_prob3 = torch.sum(det_cls_prob_product[ss_rois_num_each * 2: ss_rois_num_each * 3], 0)
            det_cls_prob4 = torch.sum(det_cls_prob_product[ss_rois_num_each * 3: ss_rois_num_each * 4], 0)
            det_cls_prob = torch.stack([det_cls_prob1, det_cls_prob2, det_cls_prob3, det_cls_prob4])
            det_cls_prob_product2 = torch.mul(cls_prob, det_prob)
        else:
            det_cls_prob = torch.sum(det_cls_prob_product, 0)
            det_cls_prob_product2 = torch.mul(cls_prob, det_prob)

        # bbox_pred = self.bbox_pred_net(fc7)
        bbox_pred = torch.zeros(cls_prob.shape[0], 80)
        self._predictions["bbox_pred"] = bbox_pred
        self._predictions['det_cls_prob'] = det_cls_prob
        self._predictions['det_cls_prob_product'] = det_cls_prob_product2

        roi_labels_1, keep_inds_1, fg_num_1, bg_num_1 = get_refine_supervision_ac4_IA(det_cls_prob_product2,
                                                                       self._image_gt_summaries['ss_boxes_input'][0],
                                                                       self._image_gt_summaries['image_level_label'],
                                                                       self._im_info)
        roi_labels_1_new = np.vstack((roi_labels_1[0], roi_labels_1[1], roi_labels_1[2], roi_labels_1[3]))
        keep_inds_1_new = np.concatenate((keep_inds_1[0], keep_inds_1[1] + pi, keep_inds_1[2] + pi*2, keep_inds_1[3] + pi*3))
        fg_num_1_new = fg_num_1[0] + fg_num_1[1] + fg_num_1[2] + fg_num_1[3]
        bg_num_1_new = bg_num_1[0] + bg_num_1[1] + bg_num_1[2] + bg_num_1[3]

        roi_labels_2, keep_inds_2, fg_num_2, bg_num_2 = get_refine_supervision_ac4_IA(refine_prob_1_new,
                                                                       self._image_gt_summaries['ss_boxes_input'][0],
                                                                       self._image_gt_summaries['image_level_label'],
                                                                       self._im_info)
        roi_labels_2_new = np.vstack((roi_labels_2[0], roi_labels_2[1], roi_labels_2[2], roi_labels_2[3]))
        keep_inds_2_new = np.concatenate((keep_inds_2[0], keep_inds_2[1] + pi, keep_inds_2[2] + pi*2, keep_inds_2[3] + pi*3))
        fg_num_2_new = fg_num_2[0] + fg_num_2[1] + fg_num_2[2] + fg_num_2[3]
        bg_num_2_new = bg_num_2[0] + bg_num_2[1] + bg_num_2[2] + bg_num_2[3]

        self.eval()
        gt = np.argmax(roi_labels_1_new, axis=1)
        mask_1 = self._inverted_attention(bbox_feats_new, gt, keep_inds_1_new, 1, step, fg_num_1_new, bg_num_1_new)
        gt = np.argmax(roi_labels_2_new, axis=1)
        mask_2 = self._inverted_attention(bbox_feats_new, gt, keep_inds_2_new, 2, step, fg_num_2_new, bg_num_2_new)
        self.train()

        mask_1 = Variable(mask_1, requires_grad=True)
        mask_2 = Variable(mask_2, requires_grad=True)
        pool5_roi_1 = pool5_roi * mask_1
        pool5_roi_2 = pool5_roi * mask_2
        fc7_roi_1 = self._head_to_tail(pool5_roi_1)
        fc7_roi_2 = self._head_to_tail(pool5_roi_2)
        refine_score_1 = self.refine_net_1(fc7_roi_1)
        refine_score_2 = self.refine_net_2(fc7_roi_2)

        refine_prob_1 = F.softmax(refine_score_1, dim=1)
        refine_prob_2 = F.softmax(refine_score_2, dim=1)
        # refine_prob_3 = F.softmax(refine_score_3, dim=1)

        self._predictions['refine_prob_1'] = refine_prob_1
        self._predictions['refine_prob_2'] = refine_prob_2
        # self._predictions['refine_prob_3'] = refine_prob_3
        return roi_labels_1, keep_inds_1, roi_labels_2, keep_inds_2, bbox_pred

    def _image_to_head(self):
        raise NotImplementedError

    def _head_to_tail(self, pool5):
        raise NotImplementedError

    def create_architecture(self, num_classes, tag=None, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        self._tag = tag
        self._num_classes = num_classes
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)
        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)
        self._num_anchors = self._num_scales * self._num_ratios
        assert tag != None
        # Initialize layers
        self._init_modules()

    def _init_modules(self):
        self._init_head_tail()
        self.cls_score_net = nn.Linear(self._fc7_channels, self._num_classes)
        self.det_score_net = nn.Linear(self._fc7_channels, self._num_classes)
        #self.bbox_pred_net = nn.Linear(self._fc7_channels, self._num_classes)
        self.bbox_pred_net = nn.Linear(self._fc7_channels, self._num_classes * 4)
        self.refine_net_1 = nn.Linear(self._fc7_channels, self._num_classes + 1)
        self.refine_net_2 = nn.Linear(self._fc7_channels, self._num_classes + 1)
        # self.refine_net_3 = nn.Linear(self._fc7_channels, self._num_classes + 1)
        # self.theta = nn.Conv2d(in_channels=512, out_channels=256,
        #                        kernel_size=1, stride=1, padding=0)
        # self.phi = nn.Conv2d(in_channels=512, out_channels=256,
        #                      kernel_size=1, stride=1, padding=0)
        # self.g = nn.Conv2d(in_channels=512, out_channels=256,
        #                      kernel_size=1, stride=1, padding=0)
        # self.W = nn.Conv2d(in_channels=256, out_channels=512,
        #                    kernel_size=1, stride=1, padding=0)
        # nn.init.constant_(self.W.weight, 0)
        # nn.init.constant_(self.W.bias, 0)
        self.init_weights()

    def _run_summary_op(self, val=False):
        """
        Run the summary operator: feed the placeholders with corresponding newtork outputs(activations)
        """
        summaries = []
        # Add image gt
        summaries.append(self._add_gt_image_summary())
        # Add event_summaries
        for key, var in self._event_summaries.items():  # __event_summaries is equal to loss itmes
            summaries.append(tb.summary.scalar(key, var.item()))
        self._event_summaries = {}
        return summaries

    def _predict_train(self, ss_boxes, step):
        torch.backends.cudnn.benchmark = False
        net_conv = self._image_to_head()
        i = 0
        rois = torch.from_numpy(ss_boxes[i]).to(self._device)
        pool5_roi_0 = self.RoIRingPool(net_conv[i][0:1, :], rois[0, :])
        pool5_context_0 = self.RoIRingPool_context(net_conv[i][0:1, :], rois[0, :])
        pool5_frame_0 = self.RoIRingPool_frame(net_conv[i][0:1, :], rois[0, :])
        pool5_roi_flip_0 = self.RoIRingPool(net_conv[i][1:2, :], rois[1, :])
        pool5_context_flip_0 = self.RoIRingPool_context(net_conv[i][1:2, :], rois[1, :])
        pool5_frame_flip_0 = self.RoIRingPool_frame(net_conv[i][1:2, :], rois[1, :])
        i = 1
        rois = torch.from_numpy(ss_boxes[i]).to(self._device)
        pool5_roi_1 = self.RoIRingPool(net_conv[i][0:1, :], rois[0, :])
        pool5_context_1 = self.RoIRingPool_context(net_conv[i][0:1, :], rois[0, :])
        pool5_frame_1 = self.RoIRingPool_frame(net_conv[i][0:1, :], rois[0, :])
        pool5_roi_flip_1 = self.RoIRingPool(net_conv[i][1:2, :], rois[1, :])
        pool5_context_flip_1 = self.RoIRingPool_context(net_conv[i][1:2, :], rois[1, :])
        pool5_frame_flip_1 = self.RoIRingPool_frame(net_conv[i][1:2, :], rois[1, :])

        pool5_roi = torch.cat((pool5_roi_0, pool5_roi_flip_0, pool5_roi_1, pool5_roi_flip_1))
        pool5_context = torch.cat((pool5_context_0, pool5_context_flip_0, pool5_context_1, pool5_context_flip_1))
        pool5_frame = torch.cat((pool5_frame_0, pool5_frame_flip_0, pool5_frame_1, pool5_frame_flip_1))

        if self._mode == 'TRAIN':
            torch.backends.cudnn.benchmark = True  # benchmark because now the input size are fixed
        fc7_roi = self._head_to_tail(pool5_roi)
        fc7_context = self._head_to_tail(pool5_context)
        fc7_frame = self._head_to_tail(pool5_frame)
        if self.ca_iw:
            rois = pool5_roi
        else:
            rois = None

        roi_labels_1, keep_inds_1, \
        roi_labels_2, keep_inds_2, bbox_pred = self._region_classification_train(pool5_roi, fc7_roi,fc7_context, fc7_frame, step)
        return roi_labels_1, keep_inds_1, roi_labels_2, keep_inds_2, bbox_pred, rois

    def _predict_test(self, ss_boxes):
        torch.backends.cudnn.benchmark = False
        net_conv = self._image_to_head()
        ss_rois = torch.from_numpy(ss_boxes).to(self._device)
        rois = ss_rois
        self._predictions["rois"] = rois
        pool5_roi = self.RoIRingPool(net_conv, rois)
        pool5_context = self.RoIRingPool_context(net_conv, rois)
        pool5_frame = self.RoIRingPool_frame(net_conv, rois)

        if self._mode == 'TRAIN':
            torch.backends.cudnn.benchmark = True  # benchmark because now the input size are fixed
        fc7_roi = self._head_to_tail(pool5_roi)
        fc7_context = self._head_to_tail(pool5_context)
        fc7_frame = self._head_to_tail(pool5_frame)

        cls_prob, det_prob, bbox_pred, cls_det_prob_product, det_cls_prob = self._region_classification_test(fc7_roi, fc7_context, fc7_frame)
        return rois, cls_prob, det_prob, bbox_pred, cls_det_prob_product, det_cls_prob

    def forward(self, image, image_level_label, im_info, gt_boxes=None, ss_boxes=None, step=None, mode='TRAIN'):
        self._image_gt_summaries['image'] = image
        self._image_gt_summaries['image_level_label'] = image_level_label
        self._image_gt_summaries['gt_boxes'] = gt_boxes
        self._image_gt_summaries['im_info'] = im_info
        self._mode = mode
        self._im_info = im_info
        self._image_level_label = torch.from_numpy(image_level_label) if image_level_label is not None else None

        if mode == 'TEST':
            self._image_gt_summaries['ss_boxes'] = ss_boxes
            self._image = torch.from_numpy(image.transpose([0, 3, 1, 2]).copy()).to(self._device)
            self._gt_boxes = torch.from_numpy(gt_boxes).to(self._device) if gt_boxes is not None else None
            self.ss_boxes_indexes = self.return_ss_boxes(np.arange(ss_boxes.shape[0]), mode)
            rois, cls_prob, det_prob, bbox_pred, cls_det_prob_product, det_cls_prob = self._predict_test(ss_boxes[self.ss_boxes_indexes, :])
            bbox_pred = bbox_pred[:, :80]
            stds = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
            means = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
            self._predictions["bbox_pred"] = bbox_pred.mul(stds).add(means)
        else:
            ss_boxes_all = []
            self._image = []
            self._image_gt_summaries['ss_boxes_input'] = []
            self._image_level_label = torch.from_numpy(image_level_label) if image_level_label is not None else None
            self.ss_boxes_indexes = self.return_ss_boxes(np.arange(ss_boxes[0].shape[0]), mode)
            for i in range(2):
                image_org = torch.from_numpy(image[i].transpose([0, 3, 1, 2]).copy()).to(self._device)
                self._image.append(image_org)
                ss_boxes_input = np.stack((ss_boxes[i * 2], ss_boxes[i * 2 + 1]))
                ss_boxes_all.append(ss_boxes_input[:, self.ss_boxes_indexes, :])
            self._image_gt_summaries['ss_boxes_input'] = ss_boxes_all

            roi_labels_1, keep_inds_1, roi_labels_2, keep_inds_2, bbox_pred, rois = self._predict_train(ss_boxes_all, step)
            bbox_pred = bbox_pred[:, :80]
            self._add_losses(roi_labels_1, keep_inds_1, roi_labels_2, keep_inds_2, step=step, rois=rois)

    def return_ss_boxes(self, boxes_index, mode='TRAIN'):
        if mode == 'TEST':
            return boxes_index
        box_num = min(1000, len(boxes_index))  # adjust the box number wrt GPU memory
        indexes = np.random.choice(boxes_index, size=box_num, replace=False)
        return indexes

    def init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

        # normal_init(self.rpn_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.rpn_cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.rpn_bbox_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.det_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.bbox_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.bbox_pred_net, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.refine_net_1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.refine_net_2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.refine_net_3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init1(self.theta, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init1(self.phi, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init1(self.g, 0, 0.01, cfg.TRAIN.TRUNCATED)

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    # only useful during testing mode
    def extract_head(self, image):
        feat = self._layers["head"](torch.from_numpy(image.transpose([0, 3, 1, 2])).to(self._device))
        return feat

    # only useful during testing mode
    def test_image(self, image, im_info, ss_boxes):
        self.eval()
        with torch.no_grad():
            self.forward(image, None, im_info, None, ss_boxes, mode='TEST')

        bbox_pred, rois, det_cls_prob, det_cls_prob_product, refine_prob_1, refine_prob_2 = \
            self._predictions['bbox_pred'].data.cpu().numpy(), \
            self._predictions['rois'].data.cpu().numpy(), \
            self._predictions['det_cls_prob'].data.cpu().numpy(), \
            self._predictions['det_cls_prob_product'].data.cpu().numpy(), \
            self._predictions['refine_prob_1'].data.cpu().numpy(), \
            self._predictions['refine_prob_2'].data.cpu().numpy()

        return bbox_pred, rois, det_cls_prob, det_cls_prob_product, refine_prob_1[:, 1:], refine_prob_2[:, 1:]

    def delete_intermediate_states(self):
        # Delete intermediate result to save memory
        for d in [self._losses, self._predictions]:
            for k in list(d):
                del d[k]

    def get_summary(self, blobs, step=None):
        self.eval()
        self.forward(blobs['data'], blobs['image_level_labels'], blobs['im_info'], blobs['gt_boxes'], blobs['ss_boxes'], step)
        self.train()
        summary = self._run_summary_op(True)
        return summary

    def train_step(self, blobs, train_op, step):
        self.forward(blobs['data'], blobs['image_level_labels'], blobs['im_info'], blobs['gt_boxes'], blobs['ss_boxes'], step)
        cls_det_loss, refine_loss_1, refine_loss_2, consistency_loss, loss = self._losses['cls_det_loss'].item(), \
                                                                             self._losses['refine_loss_1'].item(), \
                                                                             self._losses['refine_loss_2'].item(), \
                                                                             self._losses['consistency_loss'].item(), \
                                                                             self._losses['total_loss'].item()
        train_op.zero_grad()
        self._losses['total_loss'].backward()
        train_op.step()

        self.delete_intermediate_states()
        #torch.cuda.empty_cache()

        return cls_det_loss, refine_loss_1, refine_loss_2, consistency_loss, loss

    def train_step_with_summary(self, blobs, train_op, step):
        self.forward(blobs['data'], blobs['image_level_labels'], blobs['im_info'], blobs['gt_boxes'], blobs['ss_boxes'], step)
        cls_det_loss, refine_loss_1, refine_loss_2, consistency_loss, loss = self._losses["cls_det_loss"].item(), \
                                                           self._losses['refine_loss_1'].item(), \
                                                           self._losses['refine_loss_2'].item(), \
                                                           self._losses['consistency_loss'].item(), \
                                                           self._losses['total_loss'].item()
        train_op.zero_grad()
        self._losses['total_loss'].backward()
        train_op.step()
        # summary = self._run_summary_op()
        summary = 0
        self.delete_intermediate_states()
        return cls_det_loss, refine_loss_1, refine_loss_2, consistency_loss, loss, summary


    def train_step_no_return(self, blobs, train_op):
        self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], blobs['ss_boxes'])
        train_op.zero_grad()
        self._losses['total_loss'].backward()
        train_op.step()
        self.delete_intermediate_states()

    def load_state_dict(self, state_dict):
        """
        Because we remove the definition of fc layer in resnet now, it will fail when loading
        the model trained before.
        To provide back compatibility, we overwrite the load_state_dict
        """
        nn.Module.load_state_dict(self, {k: state_dict[k] for k in list(self.state_dict())})

# ----------------------------------------------------------------------------------------------------------------------
def _get_top_ranking_propoals(probs):
    """Get top ranking proposals by k-means"""
    kmeans = KMeans(n_clusters=3, random_state=3).fit(probs)
    high_score_label = np.argmax(kmeans.cluster_centers_)
    index = np.where(kmeans.labels_ == high_score_label)[0]
    if len(index) == 0:
        index = np.array([np.argmax(probs)])
    return index


def _build_graph(boxes, iou_threshold):
    """Build graph based on box IoU"""
    overlaps = bbox_overlaps(
        boxes.astype(dtype=np.float32, copy=False),
        boxes.astype(dtype=np.float32, copy=False))
    return (overlaps > iou_threshold).astype(np.float32)


def _get_graph_centers(boxes, cls_prob, im_labels):
    """Get graph centers."""
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :].copy()
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            idxs = np.where(cls_prob_tmp >= 0)[0]
            if idxs.shape[0] == 0:
                print('kmeans problem')
                continue
            idxs_tmp = _get_top_ranking_propoals(cls_prob_tmp[idxs].reshape(-1, 1))
            idxs = idxs[idxs_tmp]
            boxes_tmp = boxes[idxs, :].copy()
            cls_prob_tmp = cls_prob_tmp[idxs]
            graph = _build_graph(boxes_tmp, 0.4)

            keep_idxs = []
            gt_scores_tmp = []
            count = cls_prob_tmp.size
            while True:
                order = np.sum(graph, axis=1).argsort()[::-1]
                tmp = order[0]
                keep_idxs.append(tmp)
                inds = np.where(graph[tmp, :] > 0)[0]
                gt_scores_tmp.append(np.max(cls_prob_tmp[inds]))

                graph[:, inds] = 0
                graph[inds, :] = 0
                count = count - len(inds)
                if count <= 5:
                    break

            gt_boxes_tmp = boxes_tmp[keep_idxs, :].copy()
            gt_scores_tmp = np.array(gt_scores_tmp).copy()

            keep_idxs_new = np.argsort(gt_scores_tmp)[-1:(-1 - min(len(gt_scores_tmp), 5)):-1]

            gt_boxes = np.vstack((gt_boxes, gt_boxes_tmp[keep_idxs_new, :]))
            gt_scores = np.vstack((gt_scores,gt_scores_tmp[keep_idxs_new].reshape(-1, 1)))
            gt_classes = np.vstack((gt_classes,(i + 1) * np.ones((len(keep_idxs_new), 1), dtype=np.int32)))

            # If a proposal is chosen as a cluster center,
            # we simply delete a proposal from the candidata proposal pool,
            # because we found that the results of different strategies are similar and this strategy is more efficient
            cls_prob = np.delete(cls_prob.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)
            boxes = np.delete(boxes.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)

    proposals = {'gt_boxes': gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}

    return proposals


def _get_proposal_clusters(all_rois, proposals, im_labels):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    #gt_scores = proposals['gt_scores']
    overlaps = bbox_overlaps(
        all_rois.astype(dtype=np.float32, copy=False),
        gt_boxes.astype(dtype=np.float32, copy=False))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_labels[gt_assignment, 0]
    # cls_loss_weights = gt_scores[gt_assignment, 0]

    # # Select foreground RoIs as those with >= FG_THRESH overlap
    # fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    #
    # # Select background RoIs as those with < FG_THRESH overlap
    # bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]
    #
    # ig_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH)[0]

    # cls_loss_weights[ig_inds] = 0.0
    #
    # labels[bg_inds] = 0
    # gt_assignment[bg_inds] = -1
    #
    # img_cls_loss_weights = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    # pc_probs = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    # pc_labels = np.zeros(gt_boxes.shape[0], dtype=np.int32)
    # pc_count = np.zeros(gt_boxes.shape[0], dtype=np.int32)
    #
    # for i in xrange(gt_boxes.shape[0]):
    #     po_index = np.where(gt_assignment == i)[0]
    #     img_cls_loss_weights[i] = np.sum(cls_loss_weights[po_index])
    #     pc_labels[i] = gt_labels[i, 0]
    #     pc_count[i] = len(po_index)
    #     pc_probs[i] = np.average(cls_prob[po_index, pc_labels[i]])
    return max_overlaps, labels


def get_refine_supervision_ac4_IA(refine_prob, ss_boxes, image_level_label, im_info=None):
    '''
    refine_prob: num_box x 20 or num_box x 21
    ss_boxes; num_box x 4
    image_level_label: 1 dim vector with 20 elements
    '''
    keep_inds_list = []
    roi_labels_list = []
    pi = ss_boxes.shape[1]
    fg_len = []
    bg_len = []

    refine_prob_each = (refine_prob[0:pi, :] + refine_prob[pi:pi*2, :] + refine_prob[pi*2:pi*3, :] + refine_prob[pi*3:pi*4, :]) / 4.0
    for i in range(1):
        ss_boxes_each = ss_boxes[i, :]

        cls_prob = refine_prob_each.data.cpu().numpy()
        boxes = ss_boxes_each[:, 1:].copy()

        if refine_prob.shape[1] == image_level_label.shape[1] + 1:
            cls_prob = cls_prob[:, 1:]
        roi_labels = np.zeros([pi, image_level_label.shape[1] + 1], dtype=np.int32)
        roi_labels[:, 0] = 1  # the 0th elements is the bg
        roi_weights = np.zeros((pi, 1), dtype=np.float32)  # num_box x 1 weights of the rois

        eps = 1e-9
        cls_prob[cls_prob < eps] = eps
        cls_prob[cls_prob > 1 - eps] = 1 - eps
        proposals = _get_graph_centers(boxes.copy(), cls_prob.copy(), image_level_label.copy())
        # proposals_list.append(proposals)

        max_overlaps, labels = _get_proposal_clusters(boxes.copy(), proposals, image_level_label.copy())

        fg_inds = np.where(max_overlaps > cfg.TRAIN.MIL_FG_THRESH)[0]

        roi_labels[fg_inds, labels[fg_inds]] = 1
        roi_labels[fg_inds, 0] = 0

        bg_inds = (np.array(max_overlaps >= cfg.TRAIN.MIL_BG_THRESH_LO, dtype=np.int32) + np.array(
            max_overlaps < cfg.TRAIN.MIL_BG_THRESH_HI, dtype=np.int32) == 2).nonzero()[0]

        for m in range(4):
            if len(fg_inds) > 0 and len(bg_inds) > 0:
                fg_rois_num = min(cfg.TRAIN.MIL_NUM_FG, len(fg_inds))
                fg_inds_tmp = fg_inds[np.random.choice(np.arange(0, len(fg_inds)), size=int(fg_rois_num), replace=False)]
                bg_rois_num = min(cfg.TRAIN.MIL_NUM_BG, len(bg_inds))
                bg_inds_tmp = bg_inds[np.random.choice(np.arange(0, len(bg_inds)), size=int(bg_rois_num), replace=False)]
            elif len(fg_inds) > 0:
                fg_rois_num = min(cfg.TRAIN.MIL_NUM_FG, len(fg_inds))
                fg_inds_tmp = fg_inds[np.random.choice(np.arange(0, len(fg_inds)), size=int(fg_rois_num), replace=False)]
                bg_inds_tmp = bg_inds
            elif len(bg_inds) > 0:
                bg_rois_num = min(cfg.TRAIN.MIL_NUM_BG, len(bg_inds))
                bg_inds_tmp = bg_inds[np.random.choice(np.arange(0, len(bg_inds)), size=int(bg_rois_num), replace=False)]
                fg_inds_tmp = fg_inds
            else:
                import pdb
                pdb.set_trace()

            for n in range(1):
                keep_inds = np.concatenate([fg_inds_tmp, bg_inds_tmp])
                keep_inds_list.append(keep_inds)
                #roi_labels_list.append(roi_labels[keep_inds, :])
                roi_labels_list.append(roi_labels)
                fg_len.append(len(fg_inds_tmp))
                bg_len.append(len(bg_inds_tmp))

    return roi_labels_list, keep_inds_list, fg_len, bg_len

