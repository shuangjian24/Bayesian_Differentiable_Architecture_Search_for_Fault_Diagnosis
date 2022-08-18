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
from distribution import GaussianVariational, ScaleMixturePrior
import visdom

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--valid_set_path', type=str, default='../data/HYFJ-validwithout-noise.pkl', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.02, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--epochs', type=int, default=3, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=1, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--super_model', type=str, default='hyper-network-EXP-20201106-174803/just_train_weights-149.pt', help='path of super model')
# search-EXP-20200916-191330/just_train_weights-149.pt
# search-EXP-20200917-161950/search_phase_weights-29.pt
parser.add_argument('--super_alpha', type=str, default='evaluator-EXP-20201107-092043/plot_alpha_trend/', help='path of alpha')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--just_train', type=int, default=1, help='pure train')
parser.add_argument('--sample_num', type=int, default=1, help='posterior sample')
parser.add_argument('--epoch_flag', action='store_true', default=True, help='alpha sample flag')
parser.add_argument('--arch_infer', type=int, default=2000, help='forward inference to pick the top architecture set')
parser.add_argument('--arch_ensemble', type=int, default=10, help='child model set')
parser.add_argument('--init_alphas', action='store_true', default=True, help='init alphas')
parser.add_argument('--drop_weights_prob', type=float, default=0.3, help='drop weights probability during -just_train-')
parser.add_argument('--train_before_drop', action='store_true', default=True, help='alpha sample flag')
parser.add_argument('--path_weights_flag', action='store_true', default=False, help='alpha dropout flag')
parser.add_argument('--epoch_before_drop', type=int, default=20, help='pure train')
args = parser.parse_args()

args.save = 'sampling_arch-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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
  # print(model.classifier.weight)
  model = model.to(device)
  utils.load(model, args.super_model)
  # print(model.classifier.weight)
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  UoC_validset = pkl_to_tensorset(args.valid_set_path)
  valid_len = len(UoC_validset)
  valid_queue = torch.utils.data.DataLoader(
      UoC_validset, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(np.random.choice(range(len(UoC_validset)), valid_len)),
      pin_memory=True, num_workers=0)

  # ----------------update the alpha mu and rho------------------------
  temp_a = open(args.super_alpha + 'Alphas_normal_mu.txt', 'r', encoding='UTF-8')
  alphas_a = json.loads(temp_a.read())
  alphas_a = np.array(alphas_a)
  alphas_normal_mu = Variable(torch.from_numpy(np.float32(alphas_a[-1:, :, :])).squeeze(0).cuda(), requires_grad=False)
  model.alphas_normal_mu = alphas_normal_mu

  temp_b = open(args.super_alpha + 'Alphas_normal_rho.txt', 'r', encoding='UTF-8')
  alphas_b = json.loads(temp_b.read())
  alphas_b = np.array(alphas_b)
  alphas_normal_rho = Variable(torch.from_numpy(np.float32(alphas_b[-1:, :, :])).squeeze(0).cuda(), requires_grad=False)
  model.alphas_normal_rho = alphas_normal_rho

  temp_c = open(args.super_alpha + 'Alphas_reduce_mu.txt', 'r', encoding='UTF-8')
  alphas_c = json.loads(temp_c.read())
  alphas_c = np.array(alphas_c)
  alphas_reduce_mu = Variable(torch.from_numpy(np.float32(alphas_c[-1:, :, :])).squeeze(0).cuda(), requires_grad=False)
  model.alphas_reduce_mu = alphas_reduce_mu

  temp_d = open(args.super_alpha + 'Alphas_reduce_rho.txt', 'r', encoding='UTF-8')
  alphas_d = json.loads(temp_d.read())
  alphas_d = np.array(alphas_d)
  alphas_reduce_rho = Variable(torch.from_numpy(np.float32(alphas_d[-1:, :, :])).squeeze(0).cuda(), requires_grad=False)
  model.alphas_reduce_rho = alphas_reduce_rho
  # ---------------------------------------------------------------------
  model.normal_weight_sampler = GaussianVariational(alphas_normal_mu, alphas_normal_rho)
  model.reduce_weight_sampler = GaussianVariational(alphas_reduce_mu, alphas_reduce_rho)

  # start to inference arch
  logging.info('start to inference architecture set: sample_num %d set_num %d', args.arch_infer, args.arch_ensemble)
  viz.line([0], [-1], win='infer_acc', opts=dict(title='infer_acc'))
  viz.line([0], [-1], win='spareness', opts=dict(title='spareness'))
  viz.line([0], [-1], win='cosine', opts=dict(title='cosine'))
  viz.line([0], [-1], win='pearson', opts=dict(title='pearson'))
  geno_set = []
  arch_infer_acc_list = []
  # arch_uncertainty_metric_list = []
  arch_alpha_spareness_list = []
  alphas_similarity_cosine_list = []
  alphas_similarity_pearson_list = []
  print(model.alphas_normal_mu)
  print(model.alphas_normal_rho)
  # model._epoch_flag = True
  for i in range(args.arch_infer):
    print('==========================================================')
    inference_normal_weights_sample = model.normal_weight_sampler.sample()    # sample()
    inference_reduce_weights_sample = model.reduce_weight_sampler.sample()   # sample()
    model._get_normal_weights = inference_normal_weights_sample
    model._get_reduce_weights = inference_reduce_weights_sample

    logging.info('iter of arch_infer %d', i)
    arch_infer_acc, arch_infer_obj = arch_infer(valid_queue, model, criterion)
    logging.info('arch_infer_acc %f', arch_infer_acc)
    arch_infer_acc_list.append(arch_infer_acc)
    viz.line([arch_infer_acc], [i], win='infer_acc', update='append')

    alphas_similarity_cosine, alphas_similarity_pearson = utils.alphas_similarity(inference_normal_weights_sample, alphas_normal_mu,
                                                                                  inference_reduce_weights_sample, alphas_reduce_mu)
    logging.info('alphas_similarity_cosine %f', alphas_similarity_cosine)
    alphas_similarity_cosine_list.append(alphas_similarity_cosine)
    viz.line([alphas_similarity_cosine], [i], win='cosine', update='append')
    logging.info('alphas_similarity_pearson %f', alphas_similarity_pearson)
    alphas_similarity_pearson_list.append(alphas_similarity_pearson)
    viz.line([alphas_similarity_pearson], [i], win='pearson', update='append')

    arch_alpha_spareness = utils.alphas_sparse(inference_normal_weights_sample, inference_reduce_weights_sample)
    logging.info('alpha_spareness %f', arch_alpha_spareness)
    arch_alpha_spareness_list.append(arch_alpha_spareness)
    viz.line([arch_alpha_spareness], [i], win='spareness', update='append')
    print(model._get_normal_weights)
    print(model._get_reduce_weights)

    # logging.info('sample_normal_weights  %s', model._get_noraml_weights)
    # logging.info('sample_reduce_weights %s', model._get_reduce_weights)
    infer_geno = model.infer_genotype()
    geno_set.append(infer_geno)
    logging.info('infer_geno = %s', infer_geno)
  arch_ensamble_set_df = pd.DataFrame({'infer_acc': arch_infer_acc_list, 'infer_geno': geno_set,
                                       'alphas_similarity_cosine': alphas_similarity_cosine_list,
                                       'alphas_similarity_pearson': alphas_similarity_pearson_list,
                                       'alpha_spareness': arch_alpha_spareness_list})
  df_save_path = '../arch_inference/arch_set-' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
  arch_ensamble_set_df.to_csv(df_save_path, index=None)

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