# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys
import torch

from nets.vgg16 import MELM_vgg16

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights',
                      type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='voc_2007_trainval', type=str)
  parser.add_argument('--imdbval', dest='imdbval_name',
                      help='dataset to validate on',
                      default='voc_2007_test', type=str)
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=40000, type=int)
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152, mobile',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb


if __name__ == '__main__':
  import os
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # specify which GPU(s) to be used

  args = parse_args()
  args.set_cfgs = ['ANCHOR_SCALES', '[8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'TRAIN.STEPSIZE', '[50000]']
  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  #print('Using config:')
  #pprint.pprint(cfg)
  args.max_iters = 100000
  np.random.seed(cfg.RNG_SEED)
  torch.manual_seed(cfg.RNG_SEED)
  torch.cuda.manual_seed_all(cfg.RNG_SEED)

  # train set
  imdb, roidb = combined_roidb(args.imdb_name)
  print('{:d} roidb entries'.format(len(roidb)))

  # output directory where the models are saved
  output_dir = get_output_dir(imdb, args.tag)
  print('Output will be saved to `{:s}`'.format(output_dir))

  # tensorboard directory where the summaries are saved during training
  tb_dir = get_output_tb_dir(imdb, args.tag)
  print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  _, valroidb = combined_roidb(args.imdbval_name)
  print('{:d} validation roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip

  # load network
  if args.net == 'vgg16':
    net = MELM_vgg16()
  else:
    raise NotImplementedError
    
  train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
            pretrained_model=args.weight,
            max_iters=args.max_iters)
