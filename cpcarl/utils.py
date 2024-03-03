import torch
import torch.nn as nn
from typing import Any


def cat0(xs):
  return torch.cat(xs, dim=0)


def cat1(xs):
  return torch.cat(xs, dim=1)


def batch( features , batchsize ):
  idxs = torch.randint( low=0 , high=features.size()[0] , size=(batchsize,) )
  return features[idxs]


def MLP \
  ( features : list[ int ]
  , activations : list[ nn.Module ]
  ):

  assert (len(features) == len(activations)+1)

  i = 0
  layers = []
  while i < (len(features)-1):
    layers.append(nn.Linear(features[i], features[i+1]))
    layers.append(activations[i])
    i += 1

  layers.append(nn.Linear(features[i], 1))
  layers.append(nn.Sigmoid())

  return nn.Sequential(*layers)


# p : predicted labels
# q : true labels
def loss( p , q ):
  notp = 1.0 - p
  notq = 1.0 - q
  loglike = q * torch.log(p) + notq * torch.log(notp)
  return - torch.mean(loglike)


def reweight( nn , sources ):
  p = nn(sources)
  return (1 - p) / p

