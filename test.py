import cpcarl.utils as utils
import torch
nn = torch.nn

import numpy
from matplotlib.figure import Figure
import cpplot


NBATCHES = 4096
BATCHSIZE = 512
TESTSIZE = 1024 * 256
LR = 1e-5

  
def source(n):
  return torch.randn((n, 1))

def target(n):
  return torch.randn((n, 1)) + 1


testsourcetorch = source(TESTSIZE)
testsource = testsourcetorch.numpy()
testtarget = target(TESTSIZE).numpy()

mlp = utils.MLP([1] + [32]*3, [nn.ReLU()]*3)

optimizer = torch.optim.AdamW(mlp.parameters(), lr=LR)

testidxs = torch.randint(size=(10,), high=BATCHSIZE)

fig = Figure((6, 6))

epoch = 0
while 1:
  print("epoch:", epoch)
  sumloss = 0

  for batch in range(NBATCHES):
    sourcebatch = source(BATCHSIZE // 2)
    targetbatch = target(BATCHSIZE // 2)

    inputs = utils.cat0([sourcebatch, targetbatch])

    labels = utils.cat0 \
      ( [ torch.ones_like(sourcebatch)
        , torch.zeros_like(targetbatch)
        ]
      )

    outputs = mlp(inputs)

    l = utils.loss(outputs, labels)
    l.backward()

    optimizer.step()

    sumloss += l.item()


  print("epoch mean loss:")
  print(sumloss / NBATCHES)
  print()

  fig.clf()
  plt = fig.add_subplot(111)
  weights = utils.reweight(mlp , testsourcetorch).detach().numpy()

  binning = numpy.mgrid[-3:4:28j]
  hsource = cpplot.hist(testsource, binning, normalized=True)
  htarget = cpplot.hist(testtarget, binning, normalized=True)
  hrw = cpplot.hist(testsource, binning, weights=weights, normalized=True)

  cpplot.comparehist \
    ( plt
    , [ cpplot.zeroerr(h) for h in [ hsource , htarget , hrw ] ]
    , binning
    , [ "source" , "target" , "reweight" ]
    , "$x$"
    , "binned probability density"
    )

  plt.legend()

  fig.savefig("comp-%02d.png" % epoch)

  epoch += 1
