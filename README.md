# Chainer-NNVM-Example

[Chainer Advent Calendar 2017 - Qiita](https://qiita.com/advent-calendar/2017/chainer)の2017/12/19の記事です。 

[Chainerで学習したモデルをONNX-ChainerとNNVM/TVMを使ってAndroidへデプロイする](https://qiita.com/tkat0/items/28d1cc3b5c2831d86663)

![img](./data/top.png)

## Usage

Chainerの学習済みモデルYOLOv2_tinyをONNXに変換

```
$ cd models/YOLOv2_tiny/
$ python export.py
```

macBookのCPU,GPUにデプロイする

```
$ python run_pc.py
```

AndroidのCPU,GPUにデプロイする

```
$ python -m tvm.exec.rpc_proxy 
```

Android側でRPCプロキシサーバーに接続後、以下のコマンドを実行

```
$ python run_mobile.py
```


