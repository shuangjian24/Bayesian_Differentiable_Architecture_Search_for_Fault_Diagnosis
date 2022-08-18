import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import utils


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model

    self.criterion = nn.CrossEntropyLoss().cuda()

    self.optimizer = torch.optim.Adam([
      {"params": self.model.arch_parameters()['mu'], 'lr': 0.001},
      {"params": self.model.arch_parameters()['sigma'], "lr": 0.003}
      ], lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    theta = _concat(self.model.parameters()).data     # 将model中所有的param都展开，并拼接为一维tensor（theta即为模型所有参数）
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)        # 置零
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta   # 求梯度 = 梯度 + 参数*参数衰减率
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))   # sub替换函数：替换字符串中的某些子串（用后面的替换前面的）
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled, beta, sample_num):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid, beta, sample_num)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid, beta, sample_num):
    loss = 0
    for _ in range(sample_num):
      out_valid, KL_valid = self.model.forward_valid(input_valid)
      # print('KL_valid:', beta * KL_valid)
      alpha_spareness = torch.tensor(utils.alphas_sparse(self.model.alphas_normal_mu, self.model.alphas_reduce_mu)).cuda()
      alpha_skeda = utils.alpha_skedasticity(self.model.alphas_normal_mu) + utils.alpha_skedasticity(self.model.alphas_reduce_mu)
      temp_loss = self.criterion(out_valid, target_valid) + beta * KL_valid + 0.1*(torch.exp(-alpha_skeda) + torch.exp(-alpha_spareness))
      # print('neg likelihood:', temp_loss - beta * KL_valid)
      loss += temp_loss

    loss = loss/sample_num

    # print(loss)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()           # norm()求矩阵范数
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

