from time import time

import numpy as np
import chainer
import onnx_chainer

from PIL import Image

from model import YOLOv2_tiny, load_npz

def main():
    num_iter = 100
    chainer.config.train = False

    model = YOLOv2_tiny()

    load_npz('./YOLOtiny_v2.model', model)

    #x = np.ones((1, 3, 352, 352), dtype=np.float32) # dummy

    img = Image.open('dog.jpg')
    img = img.resize((352, 352))
    img = np.asarray(img).astype(np.float32)
    img = img.transpose(2,0,1)
    img_ = img.copy()
    img /= 255.
    img = img[np.newaxis,:]

    print('chainer benchmark start')
    start = time()
    for i in range(num_iter):
        model(img)
    end = time()

    elapsed  = (end - start) / num_iter
    print('time per forward:', elapsed)

if __name__ == '__main__':
    main()
