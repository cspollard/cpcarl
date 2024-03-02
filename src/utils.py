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

  layers = []
  for i in range(len(features)-1):
    layers.append(nn.Linear(features[i], features[i+1]))
    layers.append(activations[i])

  return nn.Sequential(*layers)


