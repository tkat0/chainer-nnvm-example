import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import chainer
from chainer import serializers
import onnx_chainer

from model import YOLOv2_tiny, load_npz, postprocess

def main():
    chainer.config.train = False

    model = YOLOv2_tiny()

    load_npz('./YOLOtiny_v2.model', model)

    x = np.ones((1, 3, 352, 352), dtype=np.float32)

    img = Image.open('000004.jpg')
    img = img.resize((352, 352))
    img = np.asarray(img).astype(np.float32)
    img = img.transpose(2,0,1)
    img_ = img.copy()
    img /= 255.
    img = img[np.newaxis,:]

    ans = model(img).data[0]
    postprocess(ans, img_)
    plt.savefig('output-chainer-darwin-cpu.png')

    print('export onnx...')
    onnx_chainer.export(model, x, filename='YOLOv2_tiny.onnx')

if __name__ == '__main__':
    main()
