import os
import sys
from time import time

import numpy as np
import onnx
import tvm
from tvm.contrib import graph_runtime, rpc
from tvm.contrib import util, ndk, rpc_proxy
import nnvm.frontend
import nnvm.compiler

from models.YOLOv2_tiny.model import postprocess

exec_gpu = False
opt_level = 0
num_iter = 100
dtype = np.float32
onnx_graph = onnx.load('./models/YOLOv2_tiny/YOLOv2_tiny.onnx')
#print(onnx_graph)
n = 352
org_img = Image.open('./data/test.jpg')
org_img = org_img.resize((n, n))
img = np.asarray(org_img).astype(np.float32).copy()
img = img.transpose(2,0,1)
img /= 255.
img = img[np.newaxis,:]

input_name = 'input_0'
data_shape = img.shape
out_shape = (1,125,n//32,n//32)

# GET model from frameworks
# change xyz to supported framework name.
sym, params = nnvm.frontend.from_onnx(onnx_graph)
#print(sym.debug_str()

# connect to the proxy
# Set to be address of tvm proxy.
proxy_host = os.environ["TVM_ANDROID_RPC_PROXY_HOST"]
proxy_port = 9090
key = "android"
print('RPC Connecting...')
remote = rpc.connect(proxy_host, proxy_port, key=key)
print('RPC Connected')

arch = "arm64"
if exec_gpu:
    # Mobile GPU
    target = 'opencl'
    target_host = "llvm -target=%s-linux-android" % arch
    ctx = remote.cl(0)
else:
    # Mobile CPU
    target = "llvm -target=%s-linux-android" % arch
    target_host = None
    ctx = remote.cpu(0)

print('Build Graph...')
with nnvm.compiler.build_config(opt_level=opt_level, add_pass=None):
    graph, lib, params = nnvm.compiler.build(sym, target, {input_name: data_shape}, params=params, target_host=target_host)
print("-------compute graph-------")
print(graph.ir())

so_name = "YOLOv2_tiny-aarch64.so"
temp = util.tempdir()
path_so = temp.relpath(so_name)
#path_so = so_name
#print(path_so)

#print('show opencl kernel')
#print(lib.imported_modules[0].get_source())

lib.export_library(path_so, ndk.create_shared)

print('DEPLOY: Shared Library Uploading...')
remote.upload(path_so)
rlib = remote.load_module(so_name)

### run on remote device
img = tvm.nd.array(img.astype(dtype), ctx)
rmodule = graph_runtime.create(graph, rlib, ctx)
rmodule.set_input('input_0', img)
rmodule.set_input(**params)
#start = time()
print('Execute')
rmodule.run()
output = tvm.nd.empty(out_shape, ctx=ctx)
output = rmodule.get_output(0, output).asnumpy()

postprocess(output[0], org_img)
plt.savefig('output-nnvm-android.png')

print('Benchmark')
ftimer = rmodule.module.time_evaluator("run", ctx, num_iter)
prof_res = ftimer()
print(prof_res)

