# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
#import random
from numpy import random

def get_minibatch(roidb, num_classes, aug=None, iter=None):
    """Given a roidb, construct a minibatch sampled from it."""
    aug_time = 2

    # Sample random scales to use for each image in this batch
    blobs = {'data': [], 'gt_boxes': [], 'im_info': [], 'image_level_labels': None, 'ss_boxes': [], 'flag': None}

    zero_index = np.arange(len(cfg.TRAIN.SCALES))
    dumy_index = npr.choice(zero_index, size=aug_time, replace=False)

    for i in range(aug_time):
        random_scale_inds = dumy_index[i]
        target_size = cfg.TRAIN.SCALES[random_scale_inds]

        # Get the input image blob, formatted for caffe
        im_blob, im_scales = _get_image_blob(roidb, random_scale_inds, aug, target_size, iter)

        blobs['data'].append(im_blob)

        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"

        blobs['im_info'].append(np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32))
        blobs['image_level_labels'] = roidb[0]['image_level_labels']

        # add ss_boxes into blob
        # Changed for WSDNN
        ss_inds = np.where(roidb[0]['gt_classes'] == -1)[0]  # remove gt_rois in ss_boxes
        ss_boxes = np.empty((len(ss_inds), 5), dtype=np.float32)
        ss_boxes[:, 1:] = roidb[0]['boxes'][ss_inds, :] * im_scales[0]
        ss_boxes[:, 0] = 0
        blobs['gt_boxes'].append(ss_boxes.copy())

        blobs['ss_boxes'].append(ss_boxes)
        ss_boxes_flip = ss_boxes.copy()
        width = im_blob.shape[2]
        num_boxes_each = ss_boxes.shape[0]
        oldx1 = ss_boxes_flip[:, 1].copy()
        oldx2 = ss_boxes_flip[:, 3].copy()
        width = np.ones((num_boxes_each)) * width
        ss_boxes_flip[:, 1] = width - oldx2
        ss_boxes_flip[:, 3] = width - oldx1
        blobs['ss_boxes'].append(ss_boxes_flip)

    return blobs


def _get_image_blob(roidb, scale_inds, aug, target_size, iter=None):
    """Builds an input blob from the images in the roidb at the specified
  scales.
  """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        if iter >= (cfg.TRAIN.STEP_ITERS - cfg.TRAIN.SNAPSHOT_ITERS):
            im, _, _ = aug(im, None, None)

        if roidb[i]['flipped']:
            im_flip = im
            im = im[:, ::-1, :]
        else:
            im_flip = im[:, ::-1, :]

        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)

        im_flip, _ = prep_im_for_blob(im_flip, cfg.PIXEL_MEANS, target_size,
                                      cfg.TRAIN.MAX_SIZE)
        processed_ims.append(im)
        processed_ims.append(im_flip)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales
