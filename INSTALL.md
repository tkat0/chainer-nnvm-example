# Install

不明点はissueやTwitterなどでご連絡ください。

## Environments

- MacBook Pro (13-inch, 2017)
    - macOS Sierra 10.12.6
- Xiaomi Mi 5
    - Android 7.0

## Setup Python environment

pyenv + anaconda 環境を利用

```
conda create -n chainer-nnvm python=3.6 anaconda
pyenv local anaconda3-5.0.0/envs/chainer-nnvm
```

```
conda install -c conda-forge protobuf
pip install -r requirements.txt
```

## Build NNVM/TVM on macOS

llvmのインストール

```
$ brew install llvm
$ export PATH=/usr/local/Cellar/llvm/HEAD-8b47f7b/bin:$PATH
```

build tvm

```
cd ./external
git clone --recursive https://github.com/dmlc/nnvm.git
git checkout cb54478af3eb061265961081426e2432a50902c9

export PYTHONPATH=/path/to/tvm/python:/path/to/tvm/topi/python:${PYTHONPATH}

cd nnvm/tvm
vim make/config.mk # change the compile options. enable OpenCL, un-commented llvm-config
make -j8
cd python
python setup.py install
cd ../topi/python
python setup.py install
```

上記`make -j8`で以下のエラーがでた

```
c++ -std=c++11 -Wall -O2 -Iinclude -I/Users/tkato/work/chainer-nnvm/nnvm/tvm/dlpack/include -I/Users/tkato/work/chainer-nnvm/nnvm/tvm/dmlc-core/include -IHalideIR/src -Itopi/include -fPIC -DTVM_CUDA_RUNTIME=0 -DTVM_ROCM_RUNTIME=0 -DTVM_OPENCL_RUNTIME=1 -DTVM_METAL_RUNTIME=0  -framework OpenCL -shared -o lib/libtvm_runtime.dylib build/runtime/c_dsl_api.o build/runtime/c_runtime_api.o build/runtime/cpu_device_api.o build/runtime/dso_module.o build/runtime/file_util.o build/runtime/module.o build/runtime/module_util.o build/runtime/registry.o build/runtime/system_lib_module.o build/runtime/thread_pool.o build/runtime/workspace_pool.o build/runtime/opencl/opencl_device_api.o build/runtime/opencl/opencl_module.o build/runtime/rpc/rpc_device_api.o build/runtime/rpc/rpc_event_impl.o build/runtime/rpc/rpc_module.o build/runtime/rpc/rpc_server_env.o build/runtime/rpc/rpc_session.o build/runtime/rpc/rpc_socket_impl.o build/runtime/graph/graph_runtime.o -pthread -lm -ldl -L/usr/local/Cellar/llvm/HEAD-8b47f7b/lib -Wl,-search_paths_first -Wl,-headerpad_max_install_names -lLLVMLTO -lLLVMPasses -lLLVMObjCARCOpts -lLLVMMIRParser -lLLVMSymbolize -lLLVMDebugInfoPDB -lLLVMDebugInfoDWARF -lLLVMCoverage -lLLVMTableGen -lLLVMDlltoolDriver -lLLVMOrcJIT -lLLVMXCoreDisassembler -lLLVMXCoreCodeGen -lLLVMXCoreDesc -lLLVMXCoreInfo -lLLVMXCoreAsmPrinter -lLLVMSystemZDisassembler -lLLVMSystemZCodeGen -lLLVMSystemZAsmParser -lLLVMSystemZDesc -lLLVMSystemZInfo -lLLVMSystemZAsmPrinter -lLLVMSparcDisassembler -lLLVMSparcCodeGen -lLLVMSparcAsmParser -lLLVMSparcDesc -lLLVMSparcInfo -lLLVMSparcAsmPrinter -lLLVMPowerPCDisassembler -lLLVMPowerPCCodeGen -lLLVMPowerPCAsmParser -lLLVMPowerPCDesc -lLLVMPowerPCInfo -lLLVMPowerPCAsmPrinter -lLLVMNVPTXCodeGen -lLLVMNVPTXDesc -lLLVMNVPTXInfo -lLLVMNVPTXAsmPrinter -lLLVMMSP430CodeGen -lLLVMMSP430Desc -lLLVMMSP430Info -lLLVMMSP430AsmPrinter -lLLVMMipsDisassembler -lLLVMMipsCodeGen -lLLVMMipsAsmParser -lLLVMMipsDesc -lLLVMMipsInfo -lLLVMMipsAsmPrinter -lLLVMLanaiDisassembler -lLLVMLanaiCodeGen -lLLVMLanaiAsmParser -lLLVMLanaiDesc -lLLVMLanaiAsmPrinter -lLLVMLanaiInfo -lLLVMHexagonDisassembler -lLLVMHexagonCodeGen -lLLVMHexagonAsmParser -lLLVMHexagonDesc -lLLVMHexagonInfo -lLLVMBPFDisassembler -lLLVMBPFCodeGen -lLLVMBPFAsmParser -lLLVMBPFDesc -lLLVMBPFInfo -lLLVMBPFAsmPrinter -lLLVMARMDisassembler -lLLVMARMCodeGen -lLLVMARMAsmParser -lLLVMARMDesc -lLLVMARMInfo -lLLVMARMAsmPrinter -lLLVMARMUtils -lLLVMAMDGPUDisassembler -lLLVMAMDGPUCodeGen -lLLVMAMDGPUAsmParser -lLLVMAMDGPUDesc -lLLVMAMDGPUInfo -lLLVMAMDGPUAsmPrinter -lLLVMAMDGPUUtils -lLLVMAArch64Disassembler -lLLVMAArch64CodeGen -lLLVMAArch64AsmParser -lLLVMAArch64Desc -lLLVMAArch64Info -lLLVMAArch64AsmPrinter -lLLVMAArch64Utils -lLLVMObjectYAML -lLLVMLibDriver -lLLVMOption -lLLVMWindowsManifest -lLLVMFuzzMutate -lLLVMX86Disassembler -lLLVMX86AsmParser -lLLVMX86CodeGen -lLLVMGlobalISel -lLLVMSelectionDAG -lLLVMAsmPrinter -lLLVMDebugInfoCodeView -lLLVMDebugInfoMSF -lLLVMX86Desc -lLLVMMCDisassembler -lLLVMX86Info -lLLVMX86AsmPrinter -lLLVMX86Utils -lLLVMMCJIT -lLLVMLineEditor -lLLVMInterpreter -lLLVMExecutionEngine -lLLVMRuntimeDyld -lLLVMCodeGen -lLLVMTarget -lLLVMCoroutines -lLLVMipo -lLLVMInstrumentation -lLLVMVectorize -lLLVMScalarOpts -lLLVMLinker -lLLVMIRReader -lLLVMAsmParser -lLLVMInstCombine -lLLVMTransformUtils -lLLVMBitWriter -lLLVMAnalysis -lLLVMProfileData -lLLVMObject -lLLVMMCParser -lLLVMMC -lLLVMBitReader -lLLVMCore -lLLVMBinaryFormat -lLLVMSupport -lLLVMDemangle -l/usr/lib/libz.dylib -lcurses -lm
clang: warning: argument unused during compilation: '-pthread' [-Wunused-command-line-argument]
ld: library not found for -l/usr/lib/libz.dylib
clang: error: linker command failed with exit code 1 (use -v to see invocation)
make: *** [lib/libtvm_runtime.dylib] Error 1
```

`-l/usr/lib/libz.dylib`を`-lz`に変えて再実行。

```
c++ -std=c++11 -Wall -O2 -Iinclude -I/Users/tkato/work/chainer-nnvm/nnvm/tvm/dlpack/include -I/Users/tkato/work/chainer-nnvm/nnvm/tvm/dmlc-core/include -IHalideIR/src -Itopi/include -fPIC -DTVM_CUDA_RUNTIME=0 -DTVM_ROCM_RUNTIME=0 -DTVM_OPENCL_RUNTIME=1 -DTVM_METAL_RUNTIME=0  -framework OpenCL -shared -o lib/libtvm_runtime.dylib build/runtime/c_dsl_api.o build/runtime/c_runtime_api.o build/runtime/cpu_device_api.o build/runtime/dso_module.o build/runtime/file_util.o build/runtime/module.o build/runtime/module_util.o build/runtime/registry.o build/runtime/system_lib_module.o build/runtime/thread_pool.o build/runtime/workspace_pool.o build/runtime/opencl/opencl_device_api.o build/runtime/opencl/opencl_module.o build/runtime/rpc/rpc_device_api.o build/runtime/rpc/rpc_event_impl.o build/runtime/rpc/rpc_module.o build/runtime/rpc/rpc_server_env.o build/runtime/rpc/rpc_session.o build/runtime/rpc/rpc_socket_impl.o build/runtime/graph/graph_runtime.o -pthread -lm -ldl -L/usr/local/Cellar/llvm/HEAD-8b47f7b/lib -Wl,-search_paths_first -Wl,-headerpad_max_install_names -lLLVMLTO -lLLVMPasses -lLLVMObjCARCOpts -lLLVMMIRParser -lLLVMSymbolize -lLLVMDebugInfoPDB -lLLVMDebugInfoDWARF -lLLVMCoverage -lLLVMTableGen -lLLVMDlltoolDriver -lLLVMOrcJIT -lLLVMXCoreDisassembler -lLLVMXCoreCodeGen -lLLVMXCoreDesc -lLLVMXCoreInfo -lLLVMXCoreAsmPrinter -lLLVMSystemZDisassembler -lLLVMSystemZCodeGen -lLLVMSystemZAsmParser -lLLVMSystemZDesc -lLLVMSystemZInfo -lLLVMSystemZAsmPrinter -lLLVMSparcDisassembler -lLLVMSparcCodeGen -lLLVMSparcAsmParser -lLLVMSparcDesc -lLLVMSparcInfo -lLLVMSparcAsmPrinter -lLLVMPowerPCDisassembler -lLLVMPowerPCCodeGen -lLLVMPowerPCAsmParser -lLLVMPowerPCDesc -lLLVMPowerPCInfo -lLLVMPowerPCAsmPrinter -lLLVMNVPTXCodeGen -lLLVMNVPTXDesc -lLLVMNVPTXInfo -lLLVMNVPTXAsmPrinter -lLLVMMSP430CodeGen -lLLVMMSP430Desc -lLLVMMSP430Info -lLLVMMSP430AsmPrinter -lLLVMMipsDisassembler -lLLVMMipsCodeGen -lLLVMMipsAsmParser -lLLVMMipsDesc -lLLVMMipsInfo -lLLVMMipsAsmPrinter -lLLVMLanaiDisassembler -lLLVMLanaiCodeGen -lLLVMLanaiAsmParser -lLLVMLanaiDesc -lLLVMLanaiAsmPrinter -lLLVMLanaiInfo -lLLVMHexagonDisassembler -lLLVMHexagonCodeGen -lLLVMHexagonAsmParser -lLLVMHexagonDesc -lLLVMHexagonInfo -lLLVMBPFDisassembler -lLLVMBPFCodeGen -lLLVMBPFAsmParser -lLLVMBPFDesc -lLLVMBPFInfo -lLLVMBPFAsmPrinter -lLLVMARMDisassembler -lLLVMARMCodeGen -lLLVMARMAsmParser -lLLVMARMDesc -lLLVMARMInfo -lLLVMARMAsmPrinter -lLLVMARMUtils -lLLVMAMDGPUDisassembler -lLLVMAMDGPUCodeGen -lLLVMAMDGPUAsmParser -lLLVMAMDGPUDesc -lLLVMAMDGPUInfo -lLLVMAMDGPUAsmPrinter -lLLVMAMDGPUUtils -lLLVMAArch64Disassembler -lLLVMAArch64CodeGen -lLLVMAArch64AsmParser -lLLVMAArch64Desc -lLLVMAArch64Info -lLLVMAArch64AsmPrinter -lLLVMAArch64Utils -lLLVMObjectYAML -lLLVMLibDriver -lLLVMOption -lLLVMWindowsManifest -lLLVMFuzzMutate -lLLVMX86Disassembler -lLLVMX86AsmParser -lLLVMX86CodeGen -lLLVMGlobalISel -lLLVMSelectionDAG -lLLVMAsmPrinter -lLLVMDebugInfoCodeView -lLLVMDebugInfoMSF -lLLVMX86Desc -lLLVMMCDisassembler -lLLVMX86Info -lLLVMX86AsmPrinter -lLLVMX86Utils -lLLVMMCJIT -lLLVMLineEditor -lLLVMInterpreter -lLLVMExecutionEngine -lLLVMRuntimeDyld -lLLVMCodeGen -lLLVMTarget -lLLVMCoroutines -lLLVMipo -lLLVMInstrumentation -lLLVMVectorize -lLLVMScalarOpts -lLLVMLinker -lLLVMIRReader -lLLVMAsmParser -lLLVMInstCombine -lLLVMTransformUtils -lLLVMBitWriter -lLLVMAnalysis -lLLVMProfileData -lLLVMObject -lLLVMMCParser -lLLVMMC -lLLVMBitReader -lLLVMCore -lLLVMBinaryFormat -lLLVMSupport -lLLVMDemangle -lz -lcurses -lm
```

つづいてNNVMのビルド

```
cd external/nnvm
make -j8
export PYTHONPATH=/path/to/nnvm/python:${PYTHONPATH}
cd python
python setup.py install
```

### Build Android RPC

まずはビルド用の環境構築

Java9のインストール。以下からダウンロード

http://www.oracle.com/technetwork/java/javase/downloads/jdk9-downloads-3848520.html

以下を.bashrcに記載

```
export JAVA_HOME=`/System/Library/Frameworks/JavaVM.framework/Versions/A/Commands/java_home -v "1.8"`
PATH=${JAVA_HOME}/bin:${PATH}
```

```
$ java --version
java 9.0.1
Java(TM) SE Runtime Environment (build 9.0.1+11)
Java HotSpot(TM) 64-Bit Server VM (build 9.0.1+11, mixed mode)
```

mavenのインストール

```
$ brew install maven
```

gradleのインストール。sdkmanを使った

```
$ curl -s "https://get.sdkman.io" | bash
$ sdk install gradle 4.4
```

Android Studioのインストール

https://developer.android.com/studio/index.html?hl=ja

インストールできたら、NDKもインストール

そして、以下を.bashrcに記載

```
export ANDROID_HOME=/Users/tkato/Library/Android/sdk
PATH=${ANDROID_HOME}/platform-tools:${PATH}
PATH=${ANDROID_HOME}/ndk-bundle:${PATH}
```

Android Standalone Toolchainを作成

```
$./make-standalone-toolchain.sh --platform=android-24 --use-llvm --arch=arm64 --install-dir=/Users/tkato/work/chainer-nnvm/external/android-toolchain-arm64
$ export TVM_NDK_CC=/Users/tkato/work/chainer-nnvm/external/android-toolchain-arm64/bin/aarch64-linux-android-g++
```

まずTVM4Jのビルド

```
cd external/nnvm/tvm
make jvmpkg
make jvminstall
```

最後に、android-rpcのビルド

https://github.com/dmlc/tvm/tree/0824de5489a6f7a9c94a804a7127d7536107edbf/apps/android_rpc

以下にでてくるconfig.mkはこんな感じ

adrenoのSDKをダウンロードして展開してパスを記載。libOpenCL.soはAndroid端末から`adb pull`で取得

OpenCLのヘッダーだけGitHubからcloneした

```
APP_ABI = arm64-v8a

APP_PLATFORM = android-24

# whether enable OpenCL during compile
USE_OPENCL = 1

# the additional include headers you want to add, e.g., SDK_PATH/adrenosdk/Development/Inc
ADD_C_INCLUDES = /Users/tkato/work/chainer-nnvm/external/adrenosdk-linux-5_0/Development/Inc/ /Users/tkato/work/chainer-nnvm/external/OpenCL-Headers/opencl20/

# the additional link libs you want to add, e.g., ANDROID_LIB_PATH/libOpenCL.so
ADD_LDLIBS = /Users/tkato/work/chainer-nnvm/external/libOpenCL.so
```

```
$ cd external/nnvm/tvm/app/android_rpc
$ vim cd apps/android_rpc/app/src/main/jni/make/config.mk
$ gradle clean build
$ ./dev_tools/gen_keystore.sh # パスワードだけ"android"。あとは適当に。
$ ./dev_tools/sign_apk.sh # パスワードに”android”と答える
```

これで、`tvm/apps/android_rpc/app/build/outputs/apk`に`tvmrpc-release.apk`が生成。これをAndroidに転送してインストール。

RPCプロキシサーバーの環境変数を通すのを忘れずに。

```
export TVM_ANDROID_RPC_PROXY_HOST=0.0.0.0
```
