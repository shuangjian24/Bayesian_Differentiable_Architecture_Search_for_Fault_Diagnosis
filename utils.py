import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import math


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):     # 重置
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):       # 更新
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1).bernoulli_(keep_prob))    # 伯努利分布，二项分布
    # mask有可能是0，有可能是1
    x.div_(keep_prob)    # x除以keep_prob, 并赋值给x
    x.mul_(mask)    # x和mask矩阵对应点乘，并赋值给x
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)



# def drop_weights(x, drop_prob):
#   if drop_prob > 0.:
#     keep_prob = 1.-drop_prob
#     MASK = torch.ones(x.size(0), x.size(1), x.size(2)).cuda()
#     for i in range(x.size(0)):
#         for j in range(x.size(1)):
#             MASK[i, j:j+1] = torch.cuda.FloatTensor(1, x.size(2)).bernoulli_(keep_prob)
#     # mask掩码矩阵
#     # x.div_(keep_prob)    # x除以keep_prob, 并赋值给x
#     # x.mul_(MASK)    # x和mask矩阵对应点乘，并赋值给x
#   return MASK

def drop_weights(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    MASK = torch.ones(x.size(0), x.size(1)).cuda()
    for i in range(x.size(0)):
        MASK[i:i+1] = torch.cuda.FloatTensor(1, x.size(1)).bernoulli_(keep_prob)
    # mask掩码矩阵
    # x.div_(keep_prob)    # x除以keep_prob, 并赋值给x
    # x.mul_(MASK)    # x和mask矩阵对应点乘，并赋值给x
  return MASK


def sparseness(x):
    # sparseness是[0，1]之间的数，值越大，说明X越稀疏
    # https://www.jianshu.com/p/e3cf4bf492eb
    _x = x.cpu().detach().numpy()
    _norm_1 = 0
    _norm_2 = 0
    for i in range(_x.shape[0]):
        temp_norm_1 = np.linalg.norm(_x[i, :], ord=1)
        _norm_1 = _norm_1 + temp_norm_1

        temp_norm_2 = np.linalg.norm(_x[i, :], ord=2)
        _norm_2 = _norm_2 + temp_norm_2
    num = _x.shape[0] * _x.shape[1]
    s2 = math.sqrt(_norm_2)
    c = _norm_1 / s2
    a = math.sqrt(num) - c
    b = math.sqrt(num) -1
    return a/b


# 度量稀疏性
def alphas_sparse(normal_sample, reduce_sample):
    alphas_sparseness = 0
    for alpha in [normal_sample, reduce_sample]:
        temp_sparseness = sparseness(alpha)
        alphas_sparseness += temp_sparseness
    return alphas_sparseness


def vector_similarity(_sample, _mu):
    '''
    将矩阵拼接为向量
    然后计算两个向量间的距离，这个距离应该是分方向的，因此，不能用平方或差的绝对值
    '''
    sample_vector = torch.cat([x.view(-1) for x in _sample]).cpu().detach().numpy()
    mu_vector = torch.cat([x.view(-1) for x in _mu]).cpu().detach().numpy()
    cosine_distance = np.dot(sample_vector, mu_vector) / (np.linalg.norm(sample_vector)*np.linalg.norm(mu_vector))   # 余弦距离

    _sample_vector = sample_vector - np.mean(sample_vector)
    _mu_vector = mu_vector - np.mean(mu_vector)
    pearson_correlation = np.dot(_sample_vector, _mu_vector) / (np.linalg.norm(_sample_vector)*np.linalg.norm(_mu_vector))     # 皮尔逊相关系数
    return cosine_distance, pearson_correlation


def alphas_similarity(normal_sample, normal_mu, reduce_sample, reduce_mu):
    normal_cosine_distance, normal_pearson_correlation = vector_similarity(normal_sample, normal_mu)
    reduce_cosine_distance, reduce_pearson_correlation = vector_similarity(reduce_sample, reduce_mu)
    return 0.75*normal_cosine_distance+0.25*reduce_cosine_distance, 0.75*normal_pearson_correlation+0.25*reduce_pearson_correlation


def alpha_skedasticity(x):
    anchor_alpha = 0.2*torch.ones(x.shape[0], x.shape[1]).cuda()
    diff_value = x - anchor_alpha
    diff_2 = diff_value**2
    return diff_2.mean()


