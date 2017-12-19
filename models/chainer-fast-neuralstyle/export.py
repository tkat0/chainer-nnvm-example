import numpy as np
import chainer
from chainer import serializers
import onnx_chainer

from PIL import Image

from model import FastStyleNet, postprocess

def main():
    chainer.config.train = False

    model = FastStyleNet()
    serializers.load_npz('./chainer-fast-neuralstyle-models/models/starrynight.model', model)

    input = './test1.jpg' 
    original = Image.open(input).convert('RGB')
    print(original.size)
    image = np.asarray(original, dtype=np.float32).transpose(2, 0, 1)
    image = image.reshape((1,) + image.shape)
    padding = 0 #50
    if padding > 0:
        image = np.pad(image, [[0, 0], [0, 0], [padding, padding], [padding, padding]], 'symmetric')
    x = image

    out = model(x)
    out = out.data[0]
    print(out.shape)
    print('model done.')

    postprocess(out)

    print('export onnx...')
    onnx_chainer.export(model, x, filename='FastStyleNet.onnx')

if __name__ == '__main__':
    main()
