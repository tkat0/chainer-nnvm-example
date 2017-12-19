import math

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable

from PIL import Image

class ResidualBlock(chainer.Chain):
    def __init__(self, n_in, n_out, stride=1, ksize=3):
        w = math.sqrt(2)
        super(ResidualBlock, self).__init__(
            c1=L.Convolution2D(n_in, n_out, ksize, stride, 1, w),
            c2=L.Convolution2D(n_out, n_out, ksize, 1, 1, w),
            b1=L.BatchNormalization(n_out),
            b2=L.BatchNormalization(n_out)
        )

    def __call__(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        if x.data.shape != h.data.shape:
            xp = chainer.cuda.get_array_module(x.data)
            n, c, hh, ww = x.data.shape
            pad_c = h.data.shape[1] - c
            p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
            p = chainer.Variable(p)
            x = F.concat((p, x))
            if x.data.shape[2:] != h.data.shape[2:]:
                x = F.average_pooling_2d(x, 1, 2)
        return h + x

class FastStyleNet(chainer.Chain):
    def __init__(self):
        super(FastStyleNet, self).__init__(
            c1=L.Convolution2D(3, 32, 9, stride=1, pad=4, nobias=True),
            c2=L.Convolution2D(32, 64, 4, stride=2, pad=1, nobias=True),
            c3=L.Convolution2D(64, 128, 4,stride=2, pad=1, nobias=True),
            r1=ResidualBlock(128, 128),
            r2=ResidualBlock(128, 128),
            r3=ResidualBlock(128, 128),
            r4=ResidualBlock(128, 128),
            r5=ResidualBlock(128, 128),
            d1=L.Deconvolution2D(128, 64, 4, stride=2, pad=1, nobias=True),
            d2=L.Deconvolution2D(64, 32, 4, stride=2, pad=1, nobias=True),
            d3=L.Deconvolution2D(32, 3, 9, stride=1, pad=4, nobias=True),
            b1=L.BatchNormalization(32),
            b2=L.BatchNormalization(64),
            b3=L.BatchNormalization(128),
            b4=L.BatchNormalization(64),
            b5=L.BatchNormalization(32),
        )

    def __call__(self, x):
        # elu to relu
        h = self.b1(F.relu(self.c1(x)))
        h = self.b2(F.relu(self.c2(h)))
        h = self.b3(F.relu(self.c3(h)))
        h = self.r1(h)
        h = self.r2(h)
        h = self.r3(h)
        h = self.r4(h)
        h = self.r5(h)
        h = self.b4(F.relu(self.d1(h)))
        h = self.b5(F.relu(self.d2(h)))
        y = self.d3(h)
        #return (F.tanh(y)+1)*127.5
        return F.tanh(y)

def postprocess(result):

    assert result.ndim == 3 and isinstance(result, np.ndarray)
    result = (result+1)*127.5

    padding = 0 #50
    median_filter = False
    keep_colors = False
    out = 'output.png'

    if padding > 0:
        result = result[:, padding:-padding, padding:-padding]
    result = np.uint8(result.transpose((1, 2, 0)))
    med = Image.fromarray(result)
    if median_filter > 0:
        print('apply median filter')
        med = med.filter(ImageFilter.MedianFilter(median_filter))
    if keep_colors:
        print('keep colors')
        med = original_colors(original, med)
    
    med.save(out)
