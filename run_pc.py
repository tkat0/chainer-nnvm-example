import sys

import numpy as np
import onnx
import tvm
from tvm.contrib import graph_runtime, rpc
import nnvm.frontend
import nnvm.compiler

from PIL import Image
from matplotlib import pyplot as plt

from models.YOLOv2_tiny.model import postprocess

num_iter = 100
dtype = np.float32
opt_level = 3

onnx_graph = onnx.load('./models/YOLOv2_tiny/YOLOv2_tiny.onnx')

org_img = Image.open('./data/test.jpg')
org_img = org_img.resize((352, 352))
img = np.asarray(org_img).astype(np.float32).copy()
img = img.transpose(2,0,1)
img /= 255.
img = img[np.newaxis,:]

org_img = np.asarray(org_img).astype(np.float32)
org_img = org_img.transpose(2,0,1)

data_array = img
input_name = 'input_0'
data_shape = data_array.shape
out_shape = (1,125,352//32,352//32)

# GET model from frameworks
# change xyz to supported framework name.
sym, params = nnvm.frontend.from_onnx(onnx_graph)
#print(sym.debug_str())

# CPU
# OPTIMIZE and COMPILE the graph to get a deployable module
# target can be "opencl", "llvm", "metal" or any target supported by tvm
target = "llvm"
with nnvm.compiler.build_config(opt_level=opt_level, add_pass=None):
    graph, lib, params = nnvm.compiler.build(sym, target, {input_name: data_shape}, params=params)

# DEPLOY and run on cpu(0)
module = graph_runtime.create(graph, lib, tvm.cpu(0))
module.set_input('input_0', tvm.nd.array(data_array.astype(dtype)))
module.set_input(**params)
module.run()

output = tvm.nd.empty(out_shape, ctx=tvm.cpu(0))
output = module.get_output(0, output).asnumpy()
#np.save('output-nnvm-darwin-cpu.npy', output)

print('benchmark on cpu')
ctx = tvm.cpu(0)
ftimer = module.module.time_evaluator("run", ctx, num_iter)
prof_res = ftimer()
print(prof_res)

# GPU
sym, params = nnvm.frontend.from_onnx(onnx_graph)

target = "opencl"
with nnvm.compiler.build_config(opt_level=opt_level, add_pass=None):
    graph, lib, params = nnvm.compiler.build(sym, target, {input_name: data_shape}, params=params)

# DEPLOY and run on gpu(0)
ctx = tvm.context(target, 0)
module = graph_runtime.create(graph, lib, ctx)
module.set_input('input_0', tvm.nd.array(data_array.astype(dtype)))
module.set_input(**params)
module.run()

output = tvm.nd.empty(out_shape, ctx=ctx)
output = module.get_output(0, output).asnumpy()

postprocess(output[0], org_img)
plt.savefig('output-nnvm-darwin-gpu.png')

print('benchmark on gpu')
ftimer = module.module.time_evaluator("run", ctx, num_iter)
prof_res = ftimer()
print(prof_res)

