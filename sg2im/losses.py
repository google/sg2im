#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F


def get_gan_losses(gan_type):
  """
  Returns the generator and discriminator loss for a particular GAN type.

  The returned functions have the following API:
  loss_g = g_loss(scores_fake)
  loss_d = d_loss(scores_real, scores_fake)
  """
  if gan_type == 'gan':
    return gan_g_loss, gan_d_loss
  elif gan_type == 'wgan':
    return wgan_g_loss, wgan_d_loss
  elif gan_type == 'lsgan':
    return lsgan_g_loss, lsgan_d_loss
  else:
    raise ValueError('Unrecognized GAN type "%s"' % gan_type)


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def _make_targets(x, y):
  """
  Inputs:
  - x: PyTorch Tensor
  - y: Python scalar

  Outputs:
  - out: PyTorch Variable with same shape and dtype as x, but filled with y
  """
  return torch.full_like(x, y)


def gan_g_loss(scores_fake):
  """
  Input:
  - scores_fake: Tensor of shape (N,) containing scores for fake samples

  Output:
  - loss: Variable of shape (,) giving GAN generator loss
  """
  if scores_fake.dim() > 1:
    scores_fake = scores_fake.view(-1)
  y_fake = _make_targets(scores_fake, 1)
  return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
  """
  Input:
  - scores_real: Tensor of shape (N,) giving scores for real samples
  - scores_fake: Tensor of shape (N,) giving scores for fake samples

  Output:
  - loss: Tensor of shape (,) giving GAN discriminator loss
  """
  assert scores_real.size() == scores_fake.size()
  if scores_real.dim() > 1:
    scores_real = scores_real.view(-1)
    scores_fake = scores_fake.view(-1)
  y_real = _make_targets(scores_real, 1)
  y_fake = _make_targets(scores_fake, 0)
  loss_real = bce_loss(scores_real, y_real)
  loss_fake = bce_loss(scores_fake, y_fake)
  return loss_real + loss_fake


def wgan_g_loss(scores_fake):
  """
  Input:
  - scores_fake: Tensor of shape (N,) containing scores for fake samples

  Output:
  - loss: Tensor of shape (,) giving WGAN generator loss
  """
  return -scores_fake.mean()


def wgan_d_loss(scores_real, scores_fake):
  """
  Input:
  - scores_real: Tensor of shape (N,) giving scores for real samples
  - scores_fake: Tensor of shape (N,) giving scores for fake samples

  Output:
  - loss: Tensor of shape (,) giving WGAN discriminator loss
  """
  return scores_fake.mean() - scores_real.mean()


def lsgan_g_loss(scores_fake):
  if scores_fake.dim() > 1:
    scores_fake = scores_fake.view(-1)
  y_fake = _make_targets(scores_fake, 1)
  return F.mse_loss(scores_fake.sigmoid(), y_fake)


def lsgan_d_loss(scores_real, scores_fake):
  assert scores_real.size() == scores_fake.size()
  if scores_real.dim() > 1:
    scores_real = scores_real.view(-1)
    scores_fake = scores_fake.view(-1)
  y_real = _make_targets(scores_real, 1)
  y_fake = _make_targets(scores_fake, 0)
  loss_real = F.mse_loss(scores_real.sigmoid(), y_real)
  loss_fake = F.mse_loss(scores_fake.sigmoid(), y_fake)
  return loss_real + loss_fake


def gradient_penalty(x_real, x_fake, f, gamma=1.0):
  N = x_real.size(0)
  device, dtype = x_real.device, x_real.dtype
  eps = torch.randn(N, 1, 1, 1, device=device, dtype=dtype)
  x_hat = eps * x_real + (1 - eps) * x_fake
  x_hat_score = f(x_hat)
  if x_hat_score.dim() > 1:
    x_hat_score = x_hat_score.view(x_hat_score.size(0), -1).mean(dim=1)
  x_hat_score = x_hat_score.sum()
  grad_x_hat, = torch.autograd.grad(x_hat_score, x_hat, create_graph=True)
  grad_x_hat_norm = grad_x_hat.contiguous().view(N, -1).norm(p=2, dim=1)
  gp_loss = (grad_x_hat_norm - gamma).pow(2).div(gamma * gamma).mean()
  return gp_loss


