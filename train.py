import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import pandas as pd
import math
from torch.autograd import Variable
from model import NetworkCIFAR as Network
# from model_search import Network
# from HYdataset_V1 import HY_train_set, HY_class_num
# from TNdataset import TN_train_set, TN_class_num
# from HYFJ_balance_dataset import HYFJ_class_num, HYFJ_dataset
from HYFJ_imbalance_noise_pkl import HYFJ_class_num, pkl_to_tensorset
import visdom

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--just_train_set_path', type=str, default='../data/HYFJ-just_trainwithout-noise.pkl', help='location of the data corpus')
parser.add_argument('--test_set_path', type=str, default='../data/HYFJ-testwithout-noise.pkl', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.03, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.002, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=2, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=1, help='num of init channels')
parser.add_argument('--layers', type=int, default=16, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = HYFJ_class_num
device = torch.device("cuda")

#################################

viz = visdom.Visdom()

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.to(device)

  genotype = eval("genotypes.%s" % args.arch)
  logging.info('genotype %s', genotype)
  # print(genotype)
  model = Network(args.init_channels*32, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  # model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion=criterion).to(device)
  model = model.to(device)

  optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay
  )

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


  # train_transform, valid_transform = utils._data_transforms_cifar10(args)
  # train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  # valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  UoC_trainset = pkl_to_tensorset(args.just_train_set_path)
  trainset_len = len(UoC_trainset)
  train_queue = torch.utils.data.DataLoader(
    UoC_trainset, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(np.random.choice(range(trainset_len), trainset_len)),
    pin_memory=True, num_workers=0
  )

  UoC_validset = pkl_to_tensorset(args.test_set_path)
  validset_len = len(UoC_validset)
  valid_queue = torch.utils.data.DataLoader(
    UoC_validset, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(np.random.choice(range(validset_len), validset_len)),
    pin_memory=True, num_workers=0
  )

  def cosineLR(max_epoch, last_epoch, min_lr, max_lr):
    temp_cosineLR = min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * last_epoch / max_epoch)) / 2
    return temp_cosineLR

  def selfdefine_lr_scheduler(epochs, max_lr, min_lr):
    max_epoch = 0.95 * epochs
    min_epoch = 0.02 * epochs
    lr_list = []
    for i in range(epochs):
      if i <= min_epoch:
        lr_list.append(min_lr)
      if min_epoch < i <= max_epoch:
        lr_list.append(cosineLR(max_epoch, i - min_epoch, min_lr, max_lr))
      if i > max_epoch:
        lr_list.append(min_lr)
    return lr_list

  LR_LIST = selfdefine_lr_scheduler(args.epochs, args.learning_rate, args.learning_rate_min)

  def adjust_learning_rate(Optimizer, epoch, lr_list):
    for param_group in Optimizer.param_groups:
      param_group['lr'] = lr_list[epoch]


  # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  # 学习率余弦退火
  best_acc = 0.0
  train_acc_list = []
  test_acc_list = []
  viz.line([[0, 0]], [-1], win='snr=free',
           opts=dict(title='snr=free: train_acc & test_acc', legend=['train_acc', 'test_acc']))
  for epoch in range(args.epochs):
    # scheduler.step()
    adjust_learning_rate(optimizer, epoch, LR_LIST)
    logging.info('epoch %d lr %e', epoch, LR_LIST[epoch])
    # logging.info('epoch %d', epoch)
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)    # top1

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)    # top1

    viz.line([[train_acc, valid_acc]], [epoch], win='snr=free', update='append')

    if valid_acc > best_acc and epoch <= int(args.epochs*0.75):
      best_acc = valid_acc
      save_model_path = 'weights-' + str(epoch) + '.pt'
      utils.save(model, os.path.join(args.save, save_model_path))
    if epoch > int(args.epochs*0.75):
      save_model_path = 'weights-' + str(epoch) + '.pt'
      utils.save(model, os.path.join(args.save, save_model_path))
    if args.epochs - epoch <= 10:
      train_acc_list.append(train_acc)
      test_acc_list.append(valid_acc)
  train_acc_mean = np.mean(np.array(train_acc_list))
  train_acc_var = np.var(np.array(train_acc_list))
  test_acc_mean = np.mean(np.array(test_acc_list))
  test_acc_var = np.var(np.array(test_acc_list))
  logging.info('train_acc_mean %f train_acc_var %f', train_acc_mean, train_acc_var)
  logging.info('test_acc_mean  %f test_acc_var %f', test_acc_mean, test_acc_var)

def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = input.to(device)      # .cuda()
    target = target.to(device)            #.cuda(device=True, non_blocking=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    # model_new, logits, genotype = model(input)      # 这个输出的genotype需要更新覆盖原来的genotype
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
    n = input.size(0)
    objs.update(loss.item(), n)       # .data[0]
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      # top1正确率：就是你预测的label取最后概率向量里面最大的那一个作为预测结果，
      # 如过你的预测结果中概率最大的那个分类正确，则预测正确。否则预测错误；
      # top5正确率：就是最后概率向量最大的前五名中，只要出现了正确概率即为预测正确。否则预测错误。
  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = input.to(device)   # .cuda()
      target = target.to(device)   # .cuda(device=True, non_blocking=True)

      logits, _ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

