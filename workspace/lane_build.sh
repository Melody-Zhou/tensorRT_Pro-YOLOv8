#! /usr/bin/bash

TRTEXEC=/home/jarvis/lean/TensorRT-8.6.1.6/bin/trtexec

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jarvis/lean/TensorRT-8.6.1.6/lib

${TRTEXEC} \
  --onnx=laneatt.sim.onnx \
  --minShapes=images:1x3x360x640 \
  --optShapes=images:1x3x360x640 \
  --maxShapes=images:8x3x360x640 \
  --memPoolSize=workspace:2048 \
  --saveEngine=laneatt.sim.FP16.trtmodel \
  --fp16 \
  > laneatt.log 2>&1