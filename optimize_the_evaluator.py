
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


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--train_set_path', type=str, default='../data/HYFJ-trainwithout-noise.pkl', help='location of the data corpus')
parser.add_argument('--valid_set_path', type=str, default='../data/HYFJ-validwithout-noise.pkl', help='location of the data corpus')
parser.add_argument('--trained_super_net', type=str, default='hyper-network-EXP-20201106-174803/just_train_weights-149.pt', help='location of the data corpus')
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
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
# just_train -> 用于训练超网络的普通权重
parser.add_argument('--just_train', type=int, default=60, help='pure train')
# sample_num -> 采样次数
parser.add_argument('--sample_num', type=int, default=1, help='posterior sample')
parser.add_argument('--epoch_flag', action='store_true', default=False, help='alpha sample flag')
parser.add_argument('--init_alphas', action='store_true', default=True, help='init alphas')
# drop_weights_prob -> 在训练超网络阶段， 按照该比例丢弃部分通道
parser.add_argument('--drop_weights_prob', type=float, default=0.2, help='drop weights probability during -just_train-')
parser.add_argument('--train_before_drop', action='store_true', default=False, help='alpha dropout flag')
parser.add_argument('--path_weights_flag', action='store_true', default=False, help='alpha dropout flag')
# epoch_before_drop -> 在训练超网络阶段，有一个预热期，在该期间不会丢弃通道
parser.add_argument('--epoch_before_drop', type=int, default=10, help='warm up train')
parser.add_argument('--arch_infer', type=int, default=30, help='forward inference to pick the top architecture set')
args = parser.parse_args()

args.save = 'evaluator-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = HYFJ_class_num
device = torch.device("cuda")


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
  utils.load(model, args.trained_super_net)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      filter(lambda p: p.requires_grad, model.parameters()),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

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

  architect = Architect(model, args)


  # ========================search=======================================
  for parameter in model.parameters():
    parameter.requires_grad = False
  Alphas_normal_mu = []
  Alphas_normal_rho = []
  Alphas_reduce_mu = []
  Alphas_reduce_rho = []

  alpha_trend_path = args.save + '/plot_alpha_trend/'
  if not os.path.exists(alpha_trend_path):
    os.mkdir(alpha_trend_path)

  logging.info('start to search the evaluator')
  model._epoch_flag = True
  model._path_weights_flag = False
  model._train_before_drop = False

  for epoch in range(args.search_epochs):
    logging.info('epoch %d', epoch)

    lr = args.learning_rate

    # search alpha
    search_alpha(train_queue, valid_queue, model, architect, optimizer, lr, epoch)

    normal_mu = model.alphas_normal_mu.cpu().detach().numpy().tolist()
    Alphas_normal_mu.append(normal_mu)
    normal_rho = model.alphas_normal_rho.cpu().detach().numpy().tolist()
    Alphas_normal_rho.append(normal_rho)

    reduce_mu = model.alphas_reduce_mu.cpu().detach().numpy().tolist()
    Alphas_reduce_mu.append(reduce_mu)
    reduce_rho = model.alphas_reduce_rho.cpu().detach().numpy().tolist()
    Alphas_reduce_rho.append(reduce_rho)

    print(model.alphas_normal_mu)
    print(model.alphas_normal_rho)

    mask_Weight = torch.ones(14, 8).cuda()
    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion, mask_Weight)
    logging.info('valid_acc %f', valid_acc)

    model_save_path = 'search_phase_weights-' + str(epoch) + '.pt'
    utils.save(model, os.path.join(args.save, model_save_path))

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    Alphas_normal_MU = json.dumps(Alphas_normal_mu)
    a = open(alpha_trend_path + "Alphas_normal_mu.txt", "w", encoding='UTF-8')
    a.write(Alphas_normal_MU)
    a.close()
    Alphas_normal_RHO = json.dumps(Alphas_normal_rho)
    b = open(alpha_trend_path + "Alphas_normal_rho.txt", "w", encoding="UTF-8")
    b.write(Alphas_normal_RHO)
    b.close()

    Alphas_reduce_MU = json.dumps(Alphas_reduce_mu)
    c = open(alpha_trend_path + "Alphas_reduce_mu.txt", "w", encoding='UTF-8')
    c.write(Alphas_reduce_MU)
    c.close()
    Alphas_reduce_RHO = json.dumps(Alphas_reduce_rho)
    d = open(alpha_trend_path + "Alphas_reduce_rho.txt", "w", encoding="UTF-8")
    d.write(Alphas_reduce_RHO)
    d.close()

  # ========================================

  # geno_set = []
  # arch_infer_acc_list = []
  # arch_alpha_spareness_list = []
  # alphas_similarity_cosine_list = []
  # alphas_similarity_pearson_list = []
  # print(model.alphas_normal_mu)
  # print(model.alphas_normal_rho)
  # for i in range(args.arch_infer):
  #   print('==========================================================')
  #   inference_normal_weights_sample = model.normal_weight_sampler.mu
  #   inference_reduce_weights_sample = model.reduce_weight_sampler.mu
  #   model._get_normal_weights = inference_normal_weights_sample
  #   model._get_reduce_weights = inference_reduce_weights_sample
  #
  #   logging.info('iter of arch_infer %d', i)
  #   arch_infer_acc, arch_infer_obj = arch_infer(valid_queue, model, criterion)
  #   logging.info('arch_infer_acc %f', arch_infer_acc)
  #   arch_infer_acc_list.append(arch_infer_acc)
    # alphas_similarity_cosine, alphas_similarity_pearson = utils.alphas_similarity(inference_normal_weights_sample, alphas_normal_mu,
    #                                                                               inference_reduce_weights_sample, alphas_reduce_mu)
    # logging.info('alphas_similarity_cosine %f', alphas_similarity_cosine)
    # alphas_similarity_cosine_list.append(alphas_similarity_cosine)
    #
    # logging.info('alphas_similarity_pearson %f', alphas_similarity_pearson)
    # alphas_similarity_pearson_list.append(alphas_similarity_pearson)

    # arch_alpha_spareness = utils.alphas_sparse(inference_normal_weights_sample, inference_reduce_weights_sample)
    # logging.info('alpha_spareness %f', arch_alpha_spareness)
    # arch_alpha_spareness_list.append(arch_alpha_spareness)
    # print(model._get_noraml_weights)
    # print(model._get_reduce_weights)

    # logging.info('sample_normal_weights  %s', model._get_noraml_weights)
    # logging.info('sample_reduce_weights = %s', model._get_reduce_weights)
  #   infer_geno = model.infer_genotype()
  #   geno_set.append(infer_geno)
  #   logging.info('infer_geno = %s', infer_geno)
  # arch_ensamble_set_df = pd.DataFrame({'infer_acc': arch_infer_acc_list, 'infer_geno': geno_set,
  #                                      'alphas_similarity_cosine': alphas_similarity_cosine_list,
  #                                      'alphas_similarity_pearson': alphas_similarity_pearson_list,
  #                                      'alpha_spareness': arch_alpha_spareness_list})
  # df_save_path = '../arch_inference/arch_set-' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
  # arch_ensamble_set_df.to_csv(df_save_path, index=None)

def search_alpha(train_queue, valid_queue, model, architect, optimizer, lr, epoch):

  m = int(len(valid_queue.dataset) / args.batch_size)
  # beta = 1/m
  model.eval()
  for step, (input, target) in enumerate(train_queue):
    beta = 1.1 ** (m - step) / ((1.1 ** m - 1)*30)

    # n = input.size(0)
    input = input.to(device)
    target = target.to(device)

    # get a random minibatch from the search queue with replacement
    # ======================更新架构超参alpha========================================
    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.to(device)
    target_search = target_search.to(device)

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled,
                   beta=beta, sample_num=args.sample_num)
    # ==================================================================================


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

def arch_infer(valid_queue, model, criterion):
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

      logits = model.forward_arch_infer(input)
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