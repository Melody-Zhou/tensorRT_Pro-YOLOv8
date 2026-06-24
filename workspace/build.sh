#! /usr/bin/bash

TRTEXEC=/home/jarvis/lean/TensorRT-8.6.1.6/bin/trtexec

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jarvis/lean/TensorRT-8.6.1.6/lib

# rfdetr (static model, TensorRT-8.6.1 FP32 required, FP16 produces incorrect confidence scores)
${TRTEXEC} --onnx=rfdetr-medium.onnx --saveEngine=rfdetr-medium.FP32.trtmodel

# rt-detr
# ${TRTEXEC} --onnx=rtdetr-l.onnx --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --saveEngine=rtdetr-l.FP32.trtmodel

# yolov10
# ${TRTEXEC} --onnx=yolov10s.onnx --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --saveEngine=yolov10s.FP32.trtmodel

# yolo26s
# ${TRTEXEC} --onnx=yolo26s.onnx --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --saveEngine=yolo26s.FP32.trtmodel
# yolo26s-seg
# ${TRTEXEC} --onnx=yolo26s-seg.onnx --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --saveEngine=yolo26s-seg.FP32.trtmodel
# yolo26s-pose
# ${TRTEXEC} --onnx=yolo26s-pose.onnx --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --saveEngine=yolo26s-pose.FP32.trtmodel
# yolo26s-obb
# ${TRTEXEC} --onnx=yolo26s-obb.onnx --minShapes=images:1x3x1024x1024 --optShapes=images:1x3x1024x1024 --maxShapes=images:16x3x1024x1024 --saveEngine=yolo26s-obb.FP32.trtmodel

# rtmo
# ${TRTEXEC} --onnx=rtmo-s_8xb32-600e_body7-640x640.onnx --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --saveEngine=rtmo-s_8xb32-600e_body7-640x640.FP32.trtmodel