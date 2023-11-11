#! /usr/bin/bash

TRTEXEC=/home/jarvis/lean/TensorRT-8.6.1.6/bin/trtexec

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jarvis/lean/TensorRT-8.6.1.6/lib

${TRTEXEC} --onnx=rtdetr-l.onnx --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --saveEngine=rtdetr-l.FP32.trtmodel
