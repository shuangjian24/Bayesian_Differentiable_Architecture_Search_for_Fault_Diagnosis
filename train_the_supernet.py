"""
train the hyper-network based on warm-up and path-dropout
step1 -> train the parametric operators respectively
step2 -> dynamic path-dropout ratio
"""
import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import json
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.utils.data as Data

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from HYFJ_imbalance_noise_pkl import HYFJ_class_num, pkl_to_tensorset
# from UoC_Dataset import UoC_class_num, UoC_DATASET
# from TNdataset_V2 import TN_train_set, TN_class_num
import visdom


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--train_set_path', type=str, default='../data/HYFJ-trainwithout-noise.pkl', help='location of the data corpus')
parser.add_argument('--valid_set_path', type=str, default='../data/HYFJ-validwithout-noise.pkl', help='location of the data corpus')
parser.add_argument('--path_trained_supernet', type=str, default='search-EXP-20200915-152047/path_weights-7.pt', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-3, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
# search_epochs -> 用于训练架构超参
parser.add_argument('--search_epochs', type=int, default=30, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=1, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
# just_train -> 用于训练超网络的普通权重
parser.add_argument('--just_train', type=int, default=150, help='pure train')
parser.add_argument('--path_train', type=int, default=20, help='pure train')
# sample_num -> 采样次数
parser.add_argument('--sample_num', type=int, default=1, help='posterior sample')
parser.add_argument('--epoch_flag', action='store_true', default=False, help='alpha sample flag')
parser.add_argument('--init_alphas', action='store_true', default=True, help='init alphas')
# drop_weights_prob -> 在训练超网络阶段， 按照该比例丢弃部分通道
parser.add_argument('--drop_weights_prob', type=float, default=0.2, help='drop weights probability during -just_train-')
parser.add_argument('--train_before_drop', action='store_true', default=False, help='alpha dropout flag')
parser.add_argument('--path_weights_flag', action='store_true', default=False, help='alpha dropout flag')
# epoch_before_drop -> 在训练超网络阶段，有一个预热期，在该期间不会丢弃通道
parser.add_argument('--epoch_before_drop', type=int, default=50, help='warm up train')
parser.add_argument('--arch_infer', type=int, default=5, help='forward inference to pick the top architecture set')
args = parser.parse_args()

args.save = 'hyper-network-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = HYFJ_class_num
device = torch.device("cuda")

viz = visdom.Visdom()

###################################3

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled = True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels*8, CIFAR_CLASSES, args.layers, criterion, epoch_flag=args.epoch_flag,
                  init_alphas=args.init_alphas, drop_alpha_prob=args.drop_weights_prob,
                  TRIAN_before_drop=args.train_before_drop, path_weights_flag=args.path_weights_flag)
  model = model.cuda()
  # utils.load(model, args.path_trained_supernet)
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      filter(lambda p: p.requires_grad, model.parameters()),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  # optimizer = torch.optim.Adam(
  #     filter(lambda p: p.requires_grad, model.parameters()),
  #     args.learning_rate,
  #     betas=(0.9, 0.99),
  #     weight_decay=args.weight_decay)

  UoC_trainset = pkl_to_tensorset(args.train_set_path)
  train_len = len(UoC_trainset)
  train_queue = torch.utils.data.DataLoader(
      UoC_trainset, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(np.random.choice(range(len(UoC_trainset)), train_len)),
      pin_memory=True, num_workers=0)

  UoC_validset = pkl_to_tensorset(args.valid_set_path)
  valid_len = len(UoC_validset)
  valid_queue = torch.utils.data.DataLoader(
      UoC_validset, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(np.random.choice(range(len(UoC_validset)), valid_len)),
      pin_memory=True, num_workers=0)

  # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
  #       optimizer, 10, eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  # the path super-net is trained over.
  logging.info('train single path step by step')
  for parameter_index in [3, 4, 5, 6, 7]:
    # print(parameter_index)
    logging.info(' parameter_index %d',  parameter_index)
    model._epoch_flag = False
    model._path_weights_flag = True
    model._train_before_drop = False

    path_weights = torch.zeros(14, 8).cuda()
    path_weights[:, parameter_index] = 0.2
    model._get_normal_weights = path_weights
    print(model._get_normal_weights)

    for name, parameter in model.named_parameters():
      if len(name.split('.')) > 7:
        if name.split('.')[5] == parameter_index:
          parameter.requires_grad = True
        else:
          parameter.requires_grad = False
      else:
          parameter.requires_grad = True


    # path_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #   optimizer, float(args.path_train), eta_min=args.learning_rate_min)

    if parameter_index == 3:
      args.path_train = 10
    else:
      args.path_train = 20

    for epoch in range(args.path_train):
      # print(epoch)
      # path_scheduler.step()
      # lr = path_scheduler.get_lr()[0]
      logging.info(' parameter_index %d epoch %d', parameter_index, epoch)
      lr = args.learning_rate
      # training
      train_acc, train_obj = train(train_queue, model, criterion, optimizer, epoch, path_weights)
      logging.info('train_acc %f', train_acc)

      # validation
      valid_acc, valid_obj = infer(valid_queue, model, criterion, path_weights)
      logging.info('valid_acc %f', valid_acc)

    model_save_path = 'path_weights-' + str(parameter_index) + '.pt'
    utils.save(model, os.path.join(args.save, model_save_path))


  logging.info('re initialize the weights of shallow layers')
  for parameter in model.parameters():
    parameter.requires_grad = True

  # for name, values in model.named_modules():
  #     # print(name)
  #     if len(name.split('.')) <= 5:
  #         if isinstance(values, nn.Conv1d):
  #             print('conv')
  #             nn.init.normal_(values.weight, 0, 0.01)
  #             if values.bias is not None:
  #                 nn.init.constant_(values.bias, 0)
  #         elif isinstance(values, nn.BatchNorm1d):
  #             print('BN')
  #             nn.init.normal_(values.weight, 1, 0.02)
  #             if values.bias is not None:
  #                 nn.init.constant_(values.bias, 0)
  #             # nn.init.constant_(values.running_mean, 0)
  #         elif isinstance(values, nn.Linear):
  #             print('Linear')
  #             nn.init.normal_(values.weight, 0, 0.01)
  #             if values.bias is not None:
  #                 nn.init.constant_(values.bias, 0)

  logging.info('train the total paths by the dropout')
  # for name, parameter in model.named_parameters():
  #     parameter.requires_grad = True
  viz.line([[0, 0]], [-1], win='snr=free',
           opts=dict(title='snr=free: train_acc & test_acc', legend=['train_acc', 'test_acc']))
  for epoch in range(args.just_train):
    '''
    total_train and dropout_train iteration
    first total_train, then dropout_train ... ...
    until convergence'''
    # scheduler.step()
    # lr = scheduler.get_lr()[0]
    logging.info('epoch %d', epoch)
    model._epoch_flag = False
    model._path_weights_flag = False

    mask_weights = utils.drop_weights(model.alphas_normal_mu, args.drop_weights_prob)
    mask_rows, mask_columns = np.where(mask_weights.cpu().numpy() == 0)
    print(mask_weights)
    if int(0.2*args.just_train) < epoch <= int(0.8*args.just_train):
        args.train_before_drop = True
        for name, parameter in model.named_parameters():
          for i in range(len(mask_rows)):
            if len(name.split('.')) >= 7:
              if name.split('.')[3] == mask_rows[i] and name.split('.')[5] == mask_columns[i]:
                parameter.requires_grad = False
              else:
                parameter.requires_grad = True
            else:
              parameter.requires_grad = True
    else:
        args.train_before_drop = False
        for name, parameter in model.named_parameters():
            parameter.requires_grad = True

    model._train_before_drop = args.train_before_drop

    # training
    train_acc, train_obj = train(train_queue, model, criterion, optimizer, epoch, mask_weights)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion, mask_weights)
    logging.info('valid_acc %f', valid_acc)

    viz.line([[train_acc, valid_acc]], [epoch], win='snr=free', update='append')

    model_save_path = 'just_train_weights-' + str(epoch) + '.pt'
    utils.save(model, os.path.join(args.save, model_save_path))


def train(train_queue, model, criterion, optimizer, epoch, mask):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  # m = int(len(train_queue.dataset) / args.batch_size)
  for step, (input, target) in enumerate(train_queue):
    # print(step)
    model.train()
    n = input.size(0)
    input = input.to(device)
    target = target.to(device)
    # 只训练普通的weight
    optimizer.zero_grad()
    logits = model.forward_just_train(input, mask)
    loss = criterion(logits, target)
    loss = loss.requires_grad_()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, mask):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  with torch.no_grad():
    model.eval()
    for step, (input, target) in enumerate(valid_queue):
      input = input.to(device)
      target = target.to(device)
      # print(input)
      # print(target)

      logits = model.forward_just_train(input, mask)
      # print(logits)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
      # print(prec1)
      # print(prec5)
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()