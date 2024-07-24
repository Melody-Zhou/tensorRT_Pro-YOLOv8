#! /usr/bin/bash

TRTEXEC=/home/jarvis/lean/TensorRT-8.6.1.6/bin/trtexec

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jarvis/lean/TensorRT-8.6.1.6/lib

${TRTEXEC} \
  --onnx=ppocr_det.sim.onnx \
  --minShapes=images:1x3x960x960 \
  --optShapes=images:1x3x960x960 \
  --maxShapes=images:8x3x960x960 \
  --memPoolSize=workspace:2048 \
  --saveEngine=ppocr_det.sim.FP16.trtmodel \
  --fp16 \
  > ppocr_det.log 2>&1

${TRTEXEC} \
  --onnx=ppocr_cls.sim.onnx \
  --minShapes=images:1x3x48x192 \
  --optShapes=images:1x3x48x192 \
  --maxShapes=images:8x3x48x192 \
  --memPoolSize=workspace:2048 \
  --saveEngine=ppocr_cls.sim.FP16.trtmodel \
  --fp16 \
  > ppocr_cls.log 2>&1

${TRTEXEC} \
  --onnx=ppocr_rec.sim.onnx \
  --minShapes=images:1x3x48x640 \
  --optShapes=images:1x3x48x640 \
  --maxShapes=images:8x3x48x640 \
  --memPoolSize=workspace:2048 \
  --saveEngine=ppocr_rec.sim.FP16.trtmodel \
  --fp16 \
  > ppocr_rec.log 2>&1