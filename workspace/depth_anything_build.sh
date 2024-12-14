#! /usr/bin/bash

TRTEXEC=/home/jarvis/lean/TensorRT-8.6.1.6/bin/trtexec

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jarvis/lean/TensorRT-8.6.1.6/lib

# V1
# ${TRTEXEC} \
#   --onnx=depth_anything_vits.sim.onnx \
#   --memPoolSize=workspace:2048 \
#   --saveEngine=depth_anything_vits.sim.FP16.trtmodel \
#   --fp16 \
#   > depth_anything_vits.log 2>&1

# V2 static
${TRTEXEC} \
  --onnx=depth_anything_v2_vits.sim.onnx \
  --memPoolSize=workspace:2048 \
  --saveEngine=depth_anything_v2_vits.sim.FP16.trtmodel \
  --fp16 \
  > depth_anything_v2_vits.static.log 2>&1

# V2 dynamic
# ${TRTEXEC} \
#   --onnx=depth_anything_v2_vits.dynamic.sim.onnx \
#   --minShapes=images:1x3x518x518 \
#   --optShapes=images:1x3x518x518 \
#   --maxShapes=images:4x3x518x518 \
#   --memPoolSize=workspace:2048 \
#   --saveEngine=depth_anything_v2_vits.dynamic.sim.FP16.trtmodel \
#   --fp16 \
#   > depth_anything_v2_vits.dynamic.log 2>&1