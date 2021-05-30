# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import math

from utils.timer import Timer
from ops.nms import nms
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv
from utils.bbox import bbox_transform_inv as bbox_transform_inv1
from utils.bbox import clip_boxes as clip_boxes1

import torch

from model.apmetric import AveragePrecisionMeter

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []
    im_shape = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)
        im_shape.append(im.shape)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors), im_shape


def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale_factors, im_shape = _get_image_blob(im)

    return blobs, im_scale_factors, im_shape


def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes

def _shuffle_boxes(boxes, w, h):
    num_ss = boxes.shape[0]
    jet_scale = np.random.uniform(0.95, 1.15, num_ss)
    jet_center_w = np.random.uniform(-0.05, 0.05, num_ss)
    jet_center_h = np.random.uniform(-0.05, 0.05, num_ss)
    ss_box_jet = boxes[:, 1:].copy()
    widths = ss_box_jet[:, 2] - ss_box_jet[:, 0] + 1.0
    heights = ss_box_jet[:, 3] - ss_box_jet[:, 1] + 1.0
    ctr_x = ss_box_jet[:, 0] + 0.5 * widths + jet_center_w * widths
    ctr_y = ss_box_jet[:, 1] + 0.5 * heights + jet_center_h * heights
    widths_new = widths * jet_scale
    heights_new = heights * jet_scale
    boxes[:, 1] = ctr_x - widths_new * 0.5
    boxes[:, 2] = ctr_y - heights_new * 0.5
    boxes[:, 3] = ctr_x + widths_new * 0.5
    boxes[:, 4] = ctr_y + heights_new * 0.5
    boxes[:, 1] = np.clip(boxes[:, 1], 0, w - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, h - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, w - 1)
    boxes[:, 4] = np.clip(boxes[:, 4], 0, h - 1)
    return boxes

def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""
    for i in range(boxes.shape[0]):
        boxes[i, :] = boxes[i, :] / scales[int(inds[i])]
    return boxes

def get_ss_boxes(roidb_i, im_scales):
    ss_inds = np.arange(1, roidb_i['gt_classes'].shape[0])
    ss_boxes = np.empty((len(ss_inds), 5), dtype=np.float32)
    ss_boxes[:, 1:] = roidb_i['boxes'][ss_inds, :] * im_scales
    ss_boxes[:, 0] = 0
    return ss_boxes

def get_flipper_boxes(ss_box, width):
    oldx1 = ss_box[:, 1].copy()
    oldx2 = ss_box[:, 3].copy()
    ss_box[:, 1] = width - oldx2 - 1
    ss_box[:, 3] = width - oldx1 - 1
    return ss_box

def get_flipper_boxes1(ss_box, width):
    oldx1 = ss_box[:, 0].copy()
    oldx2 = ss_box[:, 2].copy()
    ss_box[:, 0] = width - oldx2 - 1
    ss_box[:, 2] = width - oldx1 - 1
    return ss_box

def im_detect(net, im, roidb_i):
    process = 'mean'
    blobs, im_scales, im_shape = _get_blobs(im)
    im_blob_scales = blobs['data']

    score_list = []
    pred_box_list = []
    det_cls_list = []

    for index, scale in enumerate(im_scales):
        im_blob = im_blob_scales[index]
        ss_boxes = get_ss_boxes(roidb_i, scale)
        num_ss = ss_boxes.shape[0]
        h, w = im_shape[index][0], im_shape[index][1]
        ss_boxes_2 = ss_boxes.copy()
        ss_boxes = _shuffle_boxes(ss_boxes, w, h)
        ss_boxes_2 = _shuffle_boxes(ss_boxes_2, w, h)

        im_blob_flip = im_blob.copy()[:, ::-1, :]
        ss_boxes_flip = get_flipper_boxes(ss_boxes_2.copy(), im_blob.shape[1])
        for k in range(2):   # input and its flip
            if k % 2 == 1:
                im_blob_flip = im_blob_flip[np.newaxis, :, :, :]
                img_info = np.array([im_blob_flip.shape[1], im_blob_flip.shape[2], scale], dtype=np.float32)
                bbox_pred, rois, det_cls_prob, det_cls_prob_product, refine_prob_1, refine_prob_2, bbox_pred_1, bbox_pred_2 = net.test_image(
                    im_blob_flip, img_info, ss_boxes_flip)
                boxes = ss_boxes_flip[:, 1:5]
            else:
                im_blob = im_blob[np.newaxis, :, :, :]
                img_info = np.array([im_blob.shape[1], im_blob.shape[2], scale], dtype=np.float32)
                bbox_pred, rois, det_cls_prob, det_cls_prob_product, refine_prob_1, refine_prob_2, bbox_pred_1, bbox_pred_2 = net.test_image(
                    im_blob, img_info, ss_boxes)
                boxes = ss_boxes[:, 1:5]

            if cfg.TEST.BBOX_REG:
                box_deltas = torch.cat((bbox_pred_1, bbox_pred_2), dim=0)
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor([0.1000, 0.1000, 0.2000, 0.2000]).cuda() \
                             + torch.FloatTensor([0., 0., 0., 0.]).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * 20)
                boxes = np.concatenate((boxes, boxes), axis=0)
                pred_boxes_tmp = bbox_transform_inv1(torch.from_numpy(boxes).cuda().view(1, -1, 4), box_deltas, 1)

                if k % 2 == 1:
                    pred_boxes_tmp = get_flipper_boxes1(pred_boxes_tmp.copy(), im_blob.shape[2])
                pred_boxes_tmp = clip_boxes1(pred_boxes_tmp / scale, torch.from_numpy(np.asarray(im_shape[index])).cuda().view(1, 3), 1).cpu().numpy()[0, :]
            else:
                pred_boxes_tmp = np.tile(boxes, (1, refine_prob_1.shape[1]))

            if process == "mean":
                # scores = np.reshape((refine_prob_1 + refine_prob_2) / 2, [det_cls_prob_product.shape[0], -1])
                scores = refine_prob_1
                # pred_boxes = (pred_boxes_tmp[0:num_ss] + pred_boxes_tmp[num_ss: num_ss * 2])/2
                pred_boxes = pred_boxes_tmp[0:num_ss]
            else:
                raise ValueError

            score_list.append(scores)
            pred_box_list.append(pred_boxes)
            det_cls_list.append(det_cls_prob)

    scores = np.array(score_list).mean(axis=0)
    pred_boxes = np.array(pred_box_list).mean(axis=0)
    det_cls_prob = np.array(det_cls_list).mean(axis=0)
    target = np.reshape(roidb_i['image_level_labels'], (-1))
    return scores, pred_boxes, det_cls_prob, target


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue

            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            scores = dets[:, 4]
            inds = np.where((x2 > x1) & (y2 > y1))[0]
            dets = dets[inds, :]
            if dets == []:
                continue

            keep = nms(torch.from_numpy(dets), thresh).numpy()
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def test_net(net, imdb, roidb, weights_filename, max_per_image=100, thresh=0.):
    np.random.seed(cfg.RNG_SEED)
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    output_dir = get_output_dir(imdb, weights_filename)

    # -------------------------------------------------------
    ap_meter = AveragePrecisionMeter(difficult_examples=True)
    ap_meter.reset()
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    for i in range(num_images):
        im = cv2.imread(imdb.image_path_at(i))

        _t['im_detect'].tic()
        scores, boxes, det_cls_prob, target = im_detect(net, im, roidb[i])
        _t['im_detect'].toc()

        _t['misc'].tic()

        output = np.reshape(det_cls_prob[:], (1, -1))
        target = np.reshape(target[:], (1, -1))
        ap_meter.add(output, target)

        # skip j = 0, because it's the background class
        for j in range(0, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            # keep = nms(torch.from_numpy(cls_dets), cfg.TEST.NMS).numpy() if cls_dets.size > 0 else []
            # cls_dets = cls_dets[keep, :]
            # all_boxes[j][i] = cls_dets
            keep = nms(torch.from_numpy(cls_dets), cfg.TEST.NMS) if cls_dets.size > 0 else []
            all_boxes[j][i] = keep[0].numpy()

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        if i % 100 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, _t['im_detect'].average_time(),
                                                                _t['misc'].average_time()))

    ap = ap_meter.value().numpy()
    print('the classification AP is ')
    for index, cls in enumerate(imdb._classes):
        if cls == '__background__':
            continue
        print(('AP for {} = {:.4f}'.format(cls, ap[index])))
    print('__________________')
    map = 100 * ap.mean()
    print('the mAP is {:.4f}'.format(map))

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)


