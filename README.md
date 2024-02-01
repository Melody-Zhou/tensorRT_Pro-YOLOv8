
## 简介

该仓库基于 [shouxieai/tensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro)，并进行了调整以支持 YOLOv8 的各项任务。

* 目前已支持 YOLOv8、YOLOv8-Cls、YOLOv8-Seg、YOLOv8-OBB、YOLOv8-Pose、RT-DETR、ByteTrack 高性能推理！！！🚀🚀🚀
* 基于 tensorRT8.x，C++ 高级接口，C++ 部署，服务器/嵌入式使用

<div align=center><img src="./assets/output.jpg" width="50%" height="50%"></div>

## CSDN文章同步讲解
- 🔥 [YOLOv8推理详解及部署实现](https://blog.csdn.net/qq_40672115/article/details/134276907)
- 🔥 [YOLOv8-Cls推理详解及部署实现](https://blog.csdn.net/qq_40672115/article/details/134277392)
- 🔥 [YOLOv8-Seg推理详解及部署实现](https://blog.csdn.net/qq_40672115/article/details/134277752)
- 🔥 [YOLOv8-OBB推理详解及部署实现](https://blog.csdn.net/qq_40672115/article/details/135713830)
- 🔥 [YOLOv8-Pose推理详解及部署实现](https://blog.csdn.net/qq_40672115/article/details/134278117)
- 🔥 [RT-DETR推理详解及部署实现](https://blog.csdn.net/qq_40672115/article/details/134356250)


## Top News
- **2024/2/1**
  - 新增 MinMaxCalibrator 校准器，可以通过 `TRT::Calibrator::MinMax` 指定
  - 新增 mAP 测试使用的一些脚本文件，mAP 计算代码 copy 自 [yolov6/core/evaler.py#L231](https://github.com/meituan/YOLOv6/blob/main/yolov6/core/evaler.py#L231)
- **2024/1/21**
  - YOLOv8-OBB 支持
  - ByteTrack 支持，实现基本跟踪功能
- **2024/1/10**
  - 修复 IoU 计算 bug
- **2023/11/12**
  - RT-DETR 支持
- **2023/11/07**
  - 首次提交代码，YOLOv8 分类、检测、分割、姿态点估计任务支持

## 环境配置

该项目依赖于 cuda、cudnn、tensorRT、opencv、protobuf 库，请在 CMakeLists.txt 或 Makefile 中手动指定路径配置

* 服务器
  * CUDA >= 10.2
  * cuDNN >= 8.x
  * TensorRT >= 8.x
  * protobuf == 3.11.4
  * 软件安装请参考：[Ubuntu20.04软件安装大全](https://blog.csdn.net/qq_40672115/article/details/130255299)
* 嵌入式
  * jetpack >= 4.6
  * protobuf == 3.11.4

克隆该项目

```shell
git clone https://github.com/Melody-Zhou/tensorRT_Pro-YOLOv8.git
```

<details>
<summary>CMakeLists.txt 编译</summary>

1. 修改库文件路径

```cmake
# CMakeLists.txt 13 行, 修改 opencv 路径
set(OpenCV_DIR   "/usr/local/include/opencv4/")

# CMakeLists.txt 15 行, 修改 cuda 路径
set(CUDA_TOOLKIT_ROOT_DIR     "/usr/local/cuda-11.6")

# CMakeLists.txt 16 行, 修改 cudnn 路径
set(CUDNN_DIR    "/usr/local/cudnn8.4.0.27-cuda11.6")

# CMakeLists.txt 17 行, 修改 tensorRT 路径
set(TENSORRT_DIR "/opt/TensorRT-8.4.1.5")

# CMakeLists.txt 20 行, 修改 protobuf 路径
set(PROTOBUF_DIR "/home/jarvis/protobuf")
```
2. 编译

```shell
mkdir build
cd build
cmake ..
make -j64
```

</details>

<details>
<summary>Makefile 编译</summary>

1. 修改库文件路径

```makefile
# Makefile 4 行，修改 protobuf 路径
lean_protobuf  := /home/jarvis/protobuf

# Makefile 5 行，修改 tensorRT 路径
lean_tensor_rt := /opt/TensorRT-8.4.1.5

# Makefile 6 行，修改 cudnn 路径
lean_cudnn     := /usr/local/cudnn8.4.0.27-cuda11.6

# Makefile 7 行，修改 opencv 路径
lean_opencv    := /usr/local

# Makefile 8 行，修改 cuda 路径
lean_cuda      := /usr/local/cuda-11.6
```

2. 编译

```shell
make -j64
```

</details>

## 各项任务支持

<details>
<summary>YOLOv3支持</summary>

1. 下载 YOLOv3

```shell
git clone https://github.com/ultralytics/yolov3.git
```

2. 修改代码, 保证动态 batch

```python
# ========== export.py ==========

# yolov3/export.py第160行
# output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output0']
# if dynamic:
#     dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
#     if isinstance(model, SegmentationModel):
#         dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
#         dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
#         elif isinstance(model, DetectionModel):
#             dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
# 修改为：

output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output']            
if dynamic:
    dynamic = {'images': {0: 'batch'}}  # shape(1,3,640,640)
    if isinstance(model, SegmentationModel):
        dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
        dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
    elif isinstance(model, DetectionModel):
        dynamic['output'] = {0: 'batch'}  # shape(1,25200,85)
```

3. 导出 onnx 模型

```shell
cd yolov3
python export.py --weights=yolov3.pt --dynamic --simplify --include=onnx --opset=11
```

4. 复制模型并执行

```shell
cp yolov3/yolov3.onnx tensorRT_Pro-YOLOv8/workspace
cd tensorRT_Pro-YOLOv8

# 修改代码在 src/application/app_yolo.cpp: app_yolo 函数中, 使用 V3 的方式即可运行
# test(Yolo::Type::V3, TRT::Mode::FP32, "yolov3");

make yolo -j64
```

</details>

<details>
<summary>YOLOX支持</summary>

1. 下载 YOLOX

```shell
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
```

2. 导出 onnx 模型

```shell
cd YOLOX
export PYTHONPATH=$PYTHONPATH:.
python tools/export_onnx.py -c yolox_s.pth -f exps/default/yolox_s.py --output-name=yolox_s.onnx --dynamic --decode_in_inference
```

3. 复制模型并执行

```shell
cp YOLOX/yolox_s.onnx tensorRT_Pro-YOLOv8/workspace
cd tensorRT_Pro-YOLOv8

# 修改代码在 src/application/app_yolo.cpp: app_yolo 函数中, 使用 X 的方式即可运行
# test(Yolo::Type::X, TRT::Mode::FP32, "yolox_s");

make yolo -j64
```

</details>

<details>
<summary>YOLOv5支持</summary>

1. 下载 YOLOv5

```shell
git clone https://github.com/ultralytics/yolov5.git
```

2. 修改代码, 保证动态 batch

```python
# ========== export.py ==========

# yolov5/export.py第160行
# output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output0']
# if dynamic:
#     dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
#     if isinstance(model, SegmentationModel):
#         dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
#         dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
#         elif isinstance(model, DetectionModel):
#             dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
# 修改为：

output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output']            
if dynamic:
    dynamic = {'images': {0: 'batch'}}  # shape(1,3,640,640)
    if isinstance(model, SegmentationModel):
        dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
        dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
    elif isinstance(model, DetectionModel):
        dynamic['output'] = {0: 'batch'}  # shape(1,25200,85)
```

3. 导出 onnx 模型

```shell
cd yolov5
python export.py --weights=yolov5s.pt --dynamic --simplify --include=onnx --opset=11
```

4. 复制模型并执行

```shell
cp yolov5/yolov5s.onnx tensorRT_Pro-YOLOv8/workspace
cd tensorRT_Pro-YOLOv8

# 修改代码在 src/application/app_yolo.cpp: app_yolo 函数中, 使用 V5 的方式即可运行
# test(Yolo::Type::V5, TRT::Mode::FP32, "yolov5s");

make yolo -j64
```

</details>

<details>
<summary>YOLOv6支持</summary>

1. 下载 YOLOv6

```shell
git clone https://github.com/meituan/YOLOv6.git
```

2. 修改代码, 保证动态 batch


```python
# ========== export_onnx.py ==========

# YOLOv6/deploy/ONNX/export_onnx.py第84行
# output_axes = {
#     'outputs': {0: 'batch'},
# }
# 修改为：

output_axes = {
    'output': {0: 'batch'},
}

# YOLOv6/deploy/ONNX/export_onnx.py第106行
# torch.onnx.export(model, img, f, verbose=False, opset_version=13,
#                     training=torch.onnx.TrainingMode.EVAL,
#                     do_constant_folding=True,
#                     input_names=['images'],
#                     output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes']
#                     if args.end2end else ['outputs'],
#                     dynamic_axes=dynamic_axes)
# 修改为：

torch.onnx.export(model, img, f, verbose=False, opset_version=13,
                    training=torch.onnx.TrainingMode.EVAL,
                    do_constant_folding=True,
                    input_names=['images'],
                    output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                    if args.end2end else ['output'],
                    dynamic_axes=dynamic_axes)
```

3. 导出 onnx 模型

```shell
cd YOLOv6
python deploy/ONNX/export_onnx.py --weights yolov6s.pt --img 640 --dynamic-batch --simplify
```

4. 复制模型并执行

```shell
cp YOLOv6/yolov6s.onnx tensorRT_Pro-YOLOv8/workspace
cd tensorRT_Pro-YOLOv8

# 修改代码在 src/application/app_yolo.cpp: app_yolo 函数中, 使用 V6 的方式即可运行
# test(Yolo::Type::V6, TRT::Mode::FP32, "yolov6s");

make yolo -j64
```
</details>

<details>
<summary>YOLOv7支持</summary>

1. 下载 YOLOv7

```shell
git clone https://github.com/WongKinYiu/yolov7.git 
```

2. 导出 onnx 模型


```shell
python export.py --dynamic-batch --grid --simplify --weights=yolov7.pt
```

3. 复制模型并执行

```shell
cp yolov7/yolov7.onnx tensorRT_Pro-YOLOv8/workspace
cd tensorRT_Pro-YOLOv8

# 修改代码在 src/application/app_yolo.cpp: app_yolo 函数中, 使用 V7 的方式即可运行
# test(Yolo::Type::V7, TRT::Mode::FP32, "yolov7");

make yolo -j64
```

</details>

<details>
<summary>YOLOv8支持</summary>

1. 下载 YOLOv8

```shell
git clone https://github.com/ultralytics/ultralytics.git
```

2. 修改代码, 保证动态 batch

```python
# ========== head.py ==========

# ultralytics/nn/modules/head.py第72行，forward函数
# return y if self.export else (y, x)
# 修改为：

return y.permute(0, 2, 1) if self.export else (y, x)

# ========== exporter.py ==========

# ultralytics/engine/exporter.py第323行
# output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output0']
# dynamic = self.args.dynamic
# if dynamic:
#     dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
#     if isinstance(self.model, SegmentationModel):
#         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
#         dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
#     elif isinstance(self.model, DetectionModel):
#         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 84, 8400)
# 修改为：

output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output']
dynamic = self.args.dynamic
if dynamic:
    dynamic = {'images': {0: 'batch'}}  # shape(1,3,640,640)
    if isinstance(self.model, SegmentationModel):
        dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
        dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
    elif isinstance(self.model, DetectionModel):
        dynamic['output'] = {0: 'batch'}  # shape(1, 84, 8400)
```

3. 导出 onnx 模型, 在 ultralytics-main 新建导出文件 `export.py` 内容如下：

```python
# ========== export.py ==========
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

success = model.export(format="onnx", dynamic=True, simplify=True)
```

```shell
cd ultralytics-main
python export.py
```

4. 复制模型并执行

```shell
cp ultralytics/yolov8s.onnx tensorRT_Pro-YOLOv8/workspace
cd tensorRT_Pro-YOLOv8
make yolo -j64
```
</details>

<details>
<summary>YOLOv8-Cls支持</summary>

1. 下载 YOLOv8

```shell
git clone https://github.com/ultralytics/ultralytics.git
```

2. 修改代码, 保证动态 batch

```python
# ========== exporter.py ==========

# ultralytics/engine/exporter.py第323行
# output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output0']
# dynamic = self.args.dynamic
# if dynamic:
#     dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
#     if isinstance(self.model, SegmentationModel):
#         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
#         dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
#     elif isinstance(self.model, DetectionModel):
#         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 84, 8400)
# 修改为：

output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output']
dynamic = self.args.dynamic
if dynamic:
    dynamic = {'images': {0: 'batch'}}  # shape(1,3,640,640)
    dynamic['output'] = {0: 'batch'}
    if isinstance(self.model, SegmentationModel):
        dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
        dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
    elif isinstance(self.model, DetectionModel):
        dynamic['output'] = {0: 'batch'}  # shape(1, 84, 8400)
```

3. 导出 onnx 模型, 在 ultralytics-main 新建导出文件 `export.py` 内容如下：

```python
# ========== export.py ==========
from ultralytics import YOLO

model = YOLO("yolov8s-cls.pt")

success = model.export(format="onnx", dynamic=True, simplify=True)
```

```shell
cd ultralytics-main
python export.py
```

4. 复制模型并执行

```shell
cp ultralytics/yolov8s-cls.onnx tensorRT_Pro-YOLOv8/workspace
cd tensorRT_Pro-YOLOv8
make yolo_cls -j64
```
</details>

<details>
<summary>YOLOv8-Seg支持</summary>

1. 下载 YOLOv8

```shell
git clone https://github.com/ultralytics/ultralytics.git
```

2. 修改代码, 保证动态 batch

```python
# ========== head.py ==========

# ultralytics/nn/modules/head.py第106行，forward函数
# return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))
# 修改为：

return (torch.cat([x, mc], 1).permute(0, 2, 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))

# ========== exporter.py ==========

# ultralytics/engine/exporter.py第323行
# output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output0']
# dynamic = self.args.dynamic
# if dynamic:
#     dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
#     if isinstance(self.model, SegmentationModel):
#         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
#         dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
#     elif isinstance(self.model, DetectionModel):
#         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 84, 8400)
# 修改为：

output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output0']
dynamic = self.args.dynamic
if dynamic:
    dynamic = {'images': {0: 'batch'}}  # shape(1,3,640,640)
    if isinstance(self.model, SegmentationModel):
        dynamic['output0'] = {0: 'batch'}  # shape(1, 116, 8400)
        dynamic['output1'] = {0: 'batch'}  # shape(1,32,160,160)
    elif isinstance(self.model, DetectionModel):
        dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 84, 8400)
```

3. 导出 onnx 模型, 在 ultralytics-main 新建导出文件 `export.py` 内容如下：

```python
# ========== export.py ==========
from ultralytics import YOLO

model = YOLO("yolov8s-seg.pt")

success = model.export(format="onnx", dynamic=True, simplify=True)
```

```shell
cd ultralytics-main
python export.py
```

4. 复制模型并执行

```shell
cp ultralytics/yolov8s-seg.onnx tensorRT_Pro-YOLOv8/workspace
cd tensorRT_Pro-YOLOv8
make yolo_seg -j64
```
</details>

<details>
<summary>YOLOv8-OBB支持</summary>

1. 下载 YOLOv8

```shell
glit clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
git checkout tags/v8.1.0 -b v8.1.0
```

2. 修改代码, 保证动态 batch

```python
# ========== head.py ==========

# ultralytics/nn/modules/head.py第141行，forward函数
# return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))
# 修改为：

return torch.cat([x, angle], 1).permute(0, 2, 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

# ========== exporter.py ==========

# ultralytics/engine/exporter.py第353行
# output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output0']
# dynamic = self.args.dynamic
# if dynamic:
#     dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
#     if isinstance(self.model, SegmentationModel):
#         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
#         dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
#     elif isinstance(self.model, DetectionModel):
#         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 84, 8400)
# 修改为：

output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output']
dynamic = self.args.dynamic
if dynamic:
    dynamic = {'images': {0: 'batch'}}  # shape(1,3,640,640)
    if isinstance(self.model, SegmentationModel):
        dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
        dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
    elif isinstance(self.model, DetectionModel):
        dynamic['output'] = {0: 'batch'}  # shape(1, 84, 8400)
```

3. 导出 onnx 模型, 在 ultralytics-main 新建导出文件 `export.py` 内容如下：

```python
# ========== export.py ==========
from ultralytics import YOLO

model = YOLO("yolov8s-obb.pt")

success = model.export(format="onnx", dynamic=True, simplify=True)
```

```shell
cd ultralytics-main
python export.py
```

4. 复制模型并执行

```shell
cp ultralytics/yolov8s-obb.onnx tensorRT_Pro-YOLOv8/workspace
cd tensorRT_Pro-YOLOv8
make yolo_obb -j64
```

</details>

<details>
<summary>YOLOv8-Pose支持</summary>

1. 下载 YOLOv8

```shell
git clone https://github.com/ultralytics/ultralytics.git
```

2. 修改代码, 保证动态 batch

```python
# ========== head.py ==========

# ultralytics/nn/modules/head.py第130行，forward函数
# return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))
# 修改为：

return torch.cat([x, pred_kpt], 1).permute(0, 2, 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

# ========== exporter.py ==========

# ultralytics/engine/exporter.py第323行
# output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output0']
# dynamic = self.args.dynamic
# if dynamic:
#     dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
#     if isinstance(self.model, SegmentationModel):
#         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
#         dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
#     elif isinstance(self.model, DetectionModel):
#         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 84, 8400)
# 修改为：

output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output']
dynamic = self.args.dynamic
if dynamic:
    dynamic = {'images': {0: 'batch'}}  # shape(1,3,640,640)
    dynamic['output'] = {0: 'batch'}
    if isinstance(self.model, SegmentationModel):
        dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
        dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
    elif isinstance(self.model, DetectionModel):
        dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 84, 8400)
```

3. 导出 onnx 模型, 在 ultralytics-main 新建导出文件 `export.py` 内容如下：

```python
# ========== export.py ==========
from ultralytics import YOLO

model = YOLO("yolov8s-pose.pt")

success = model.export(format="onnx", dynamic=True, simplify=True)
```

```shell
cd ultralytics-main
python export.py
```

4. 复制模型并执行

```shell
cp ultralytics/yolov8s-pose.onnx tensorRT_Pro-YOLOv8/workspace
cd tensorRT_Pro-YOLOv8
make yolo_pose -j64
```
</details>

<details>
<summary>RT-DETR支持</summary>

1. 前置条件

- **tensorRT >= 8.6**

2. 下载 YOLOv8

```shell
git clone https://github.com/ultralytics/ultralytics.git
```

3. 修改代码, 保证动态 batch

```python
# ========== exporter.py ==========

# ultralytics/engine/exporter.py第323行
# output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output0']
# dynamic = self.args.dynamic
# if dynamic:
#     dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
#     if isinstance(self.model, SegmentationModel):
#         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
#         dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
#     elif isinstance(self.model, DetectionModel):
#         dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 84, 8400)
# 修改为：

output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output']
dynamic = self.args.dynamic
if dynamic:
    dynamic = {'images': {0: 'batch'}}  # shape(1,3,640,640)
    if isinstance(self.model, SegmentationModel):
        dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
        dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
    elif isinstance(self.model, DetectionModel):
        dynamic['output'] = {0: 'batch'}  # shape(1, 84, 8400)
```

4. 导出 onnx 模型，在 ultralytics-main 新建导出文件 `export.py` 内容如下（可能会由于 torch 版本问题导出失败, 具体可参考 [#6144](https://github.com/ultralytics/ultralytics/issues/6144)）

```python
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")

success = model.export(format="onnx", dynamic=True, simplify=True)
```

```shell
cd ultralytics-main
python export.py
```

5. engine 生成

- **方案一**：替换 tensorRT_Pro-YOLOv8 中的 onnxparser 解析器，具体可参考文章：[RT-DETR推理详解及部署实现](https://blog.csdn.net/qq_40672115/article/details/134356250)
- **方案二**：利用 **trtexec** 工具生成 engine

```shell
cp ultralytics/yolov8s.onnx tensorRT_Pro-YOLOv8/workspace
cd tensorRT_Pro-YOLOv8/workspace
bash build.sh
```

6. 执行

```shell
make rtdetr -j64
```

</details>


<details>
<summary>ByteTrack支持</summary>

1. 说明

代码 copy 自：[https://github.com/CYYAI/AiInfer/tree/main/utils/tracker/ByteTracker](https://github.com/CYYAI/AiInfer/tree/main/utils/tracker/ByteTracker)

以 YOLOv8 作为检测器实现基本跟踪功能（其它检测器也行）

2. demo 演示

```shell
cd tensorRT_Pro-YOLOv8
make bytetrack -j64
```

</details>

## 接口介绍

<details>
<summary>编译接口</summary>

```cpp
TRT::compile(
    mode,                       // FP32、FP16、INT8
    test_batch_size,            // max batch size
    onnx_file,                  // source 
    model_file,                 // save to
    {},                         // redefine the input shape
    int8process,                // the recall function for calibration
    "inference",                // the dir where the image data is used for calibration
    ""                          // the dir where the data generated from calibration is saved(a.k.a where to load the calibration data.)
);
```
* tensorRT_Pro 原编译接口, 支持 FP32、FP16、INT8 编译
* 模型的编译工作也可以通过 `trtexec` 工具完成
</details>

<details>
<summary>推理接口</summary>

```cpp
// 创建推理引擎在 0 号显卡上
auto engine = YoloPose::create_infer(
    engine_file,                    // engine file
    deviceid,                       // gpu id
    0.25f,                          // confidence threshold
    0.45f,                          // nms threshold
    YoloPose::NMSMethod::FastGPU,   // NMS method, fast GPU / CPU
    1024,                           // max objects
    false                           // preprocess use multi stream
);

// 加载图像
auto image = cv::imread("inference/car.jpg");

// 推理并获取结果
auto boxes = engine->commit(image).get()  // 得到的是 vector<Box>
```

</details>

## 参考
- [https://github.com/shouxieai/tensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro)
- [https://github.com/shouxieai/infer](https://github.com/shouxieai/infer)