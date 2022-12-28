# built by zhangdy    
生成可执行二进制文件 onnx_test    
g++ onnx_test.cpp -o onnx_test \
-I /workspace/mgtvdeploy_sdk/third_libs/install/onnxruntime/include/ \
-I /workspace/mgtvdeploy_sdk/third_libs/install/opencv/include/ \
-L /workspace/mgtvdeploy_sdk/third_libs/install/opencv/lib64/  \
-L /workspace/mgtvdeploy_sdk/third_libs/install/onnxruntime/lib \
-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_video -lopencv_videoio -lopencv_imgcodecs \
-lonnxruntime
#############---------------------------------------------------------------------------------------------------

# Repo contents:
* `onnx_test.cpp`: main C++ file
* `CMakeLists.txt`
* `flake.nix` and `default.nix` for reproducible builds
* `build.sh` for non-nix builds
* `cat_small.jpg` original input file
* `cat_sr.png` C++ ONNX output
* `cat_sr_pytorch.png` PyTorch and Python ONNX output

# Build instructions
### If you have Nix
```
nix build
# or, if you don't like flakes:
nix-build
```

You will find the executable in `./result/bin/onnx_test`

### Build normally
Dependencies:
* ONNX
* OpenCV

You can use `build.sh` which just contains:
```bash
# create build dir if it does not exist
[ -d build ] || mkdir build
# generate builder
cmake . -B build
# build
cmake --build build
```

# Run instructions
```
<executable_name> <onnx-model> <source-img> <dest-img>
```
for example:
```
<executable_name> bsrgan-pretrained.onnx cat_small.jpg cat_sr.png
```
