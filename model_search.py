import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
from distribution import GaussianVariational, ScaleMixturePrior
from utils import drop_weights

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, True)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm1d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
      # （4， 4， 16， 16， 1）
    super(Cell, self).__init__()
    self.reduction = reduction
    print(C_prev_prev, C_prev, C)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)

    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)
    return torch.cat(states[-self._multiplier:], dim=1)

class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, epoch_flag, init_alphas, drop_alpha_prob,
               TRIAN_before_drop, path_weights_flag, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C     # 值为8
    self._num_classes = num_classes  # 分类数
    self._layers = layers            # 值为8
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._epoch_flag = epoch_flag
    self._init_alphas = init_alphas
    self._drop_alpha_prob = drop_alpha_prob
    self._train_before_drop = TRIAN_before_drop
    self._path_weights_flag = path_weights_flag

    C_curr = stem_multiplier*C       # C_curr = 16*8 = 128
    self.stem = nn.Sequential(
      nn.Conv1d(1, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm1d(C_curr, affine=False)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C    # 128, 128, 8
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)

      reduction_prev = reduction
      self.cells += [cell]

      C_prev_prev, C_prev = C_prev, multiplier*C_curr     # 96， 4*16


    self.global_pooling = nn.AdaptiveAvgPool1d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    if self._init_alphas:
        self._initialize_alphas()

    self.normal_weight_sampler = GaussianVariational(self.alphas_normal_mu, self.alphas_normal_rho)
    self.reduce_weight_sampler = GaussianVariational(self.alphas_reduce_mu, self.alphas_reduce_rho)

    self.prior_dist = ScaleMixturePrior(pi=0.2, sigma1=0.05, sigma2=0.03, dist=None)

    self._get_normal_weights = torch.empty((14, 8)).cuda()
    self._get_reduce_weights = torch.empty((14, 8)).cuda()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion,
                        self._epoch_falg, self._init_alphas, self._drop_alpha_prob,
                        self._train_before_drop, self._path_weights_flag).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward_valid(self, input):
    self.KL_normal = 0
    self.KL_reduce = 0
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      # print(i)
      if self._epoch_flag:
        if cell.reduction:
          weights = self.reduce_weight_sampler.sample()
          self._get_reduce_weights = weights
          # print(weights)
          # print(weights.shape)
          reduce_variational_posterior = self.reduce_weight_sampler.log_posterior()
          reduce_log_prior = self.prior_dist.log_prior(weights)
          self.KL_reduce += reduce_variational_posterior - reduce_log_prior

        else:
          weights = self.normal_weight_sampler.sample()
          self._get_normal_weights = weights
          # print(weights)
          # print(weights.shape)
          normal_variational_posterior = self.normal_weight_sampler.log_posterior()
          normal_log_prior = self.prior_dist.log_prior(weights)
          self.KL_normal += normal_variational_posterior - normal_log_prior
      else:
          weights = 0.4*torch.sigmoid(torch.zeros(14, 8).cuda())
      # print(weights)
      s0, s1 = s1, cell(s0, s1, weights)
    KL_valid = (6/8)*self.KL_normal + (2/8)*self.KL_reduce
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, KL_valid

  def forward_just_train(self, input, MASK_WEIGHTS):
    '''只训练，更新普通权重'''
    s0 = s1 = self.stem(input)
    mask_weights = MASK_WEIGHTS
    for i, cell in enumerate(self.cells):
      if self._epoch_flag:
        if cell.reduction:
          weights = self.reduce_weight_sampler.mu
        else:
          weights = self.normal_weight_sampler.mu
      else:
          weights = 0.4*torch.sigmoid(torch.zeros(14, 8).cuda())
          if self._path_weights_flag:
            weights = self._get_normal_weights
          if self._train_before_drop:
            weights = weights.mul_(mask_weights)
      # print(weights)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits

  def forward_arch_infer(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = self._get_reduce_weights
      else:
        weights = self._get_normal_weights
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal_mu = Variable(0.2*torch.ones(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce_mu = Variable(0.2*torch.ones(k, num_ops).cuda(), requires_grad=True)
    self.alphas_normal_rho = Variable(-4*torch.ones(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce_rho = Variable(-4*torch.ones(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = dict(mu=[self.alphas_normal_mu, self.alphas_reduce_mu],
                                 sigma=[self.alphas_normal_rho, self.alphas_reduce_rho])

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]

        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(self.alphas_normal_mu.data.cpu().numpy())
    # # self.normal_weight_sampler.mu
    gene_reduce = _parse(self.alphas_reduce_mu.data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def infer_genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(self._get_normal_weights.data.cpu().numpy())
    # # self.normal_weight_sampler.mu
    gene_reduce = _parse(self._get_reduce_weights.data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


def main():
  device = torch.device("cuda")
  genotype = eval("genotypes.%s" % "DARTS")
  model = Network(4, 3, 10, False, genotype).to(device)
  x = torch.randn(2, 1, 64).to(device)
  out = model(x)
  print(out)

if __name__ == '__main__':
    main()
