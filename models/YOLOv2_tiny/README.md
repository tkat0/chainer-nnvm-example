# YOLOv2_tiny

Chainerの学習済みモデルをONNXフォーマットでエクスポートする

モデル定義と学習済みモデルは[leetenki/YOLOtiny_v2_chainer](https://github.com/leetenki/YOLOtiny_v2_chainer/)を利用させていただきました。

## download trained model

```
$ wget https://github.com/leetenki/YOLOtiny_v2_chainer/raw/master/YOLOtiny_v2.model
```

## export ONNX

```
$ python export.py
```

## benchmark

```
$ python benchmark.py
```
