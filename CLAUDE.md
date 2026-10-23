# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

### Makefile (primary)

1. Edit `Makefile` to set library paths for your environment:
   - `lean_tensor_rt` — TensorRT install path
   - `lean_cudnn` — cuDNN install path
   - `lean_cuda` — CUDA install path
   - `lean_opencv` — OpenCV install path
   - `cuda_arch` — GPU SM architecture (e.g., `-gencode=arch=compute_75,code=sm_75`)
2. `make -j$(nproc)` → produces `workspace/pro`

### CMakeLists.txt (alternative)

1. Edit `CMakeLists.txt` paths
2. `mkdir build && cd build && cmake .. && make -j$(nproc)`

### Running a model

```shell
cd workspace && ./pro <task_name>
# e.g.: ./pro yolo, ./pro yolo_pose, ./pro bytetrack, ./pro rtmo, ./pro ppocr, ./pro yolo_depth
```

All available tasks are listed in `src/main.cpp` dispatch. The Makefile provides convenience targets: `make yolo`, `make yolo_pose`, `make depth_anything`, `make yolo_depth`, etc.

### Engine generation via trtexec

For models requiring `trtexec` (RT-DETR, RTMO, lane detection, Depth-Anything, YOLOv10, YOLO26 detection variants), use shell scripts in `workspace/` (e.g., `build.sh`, `lane_build.sh`, `depth_anything_build.sh`, `ocr_build.sh`).

YOLO26-Sem and YOLO26-Depth are exceptions: their entry points auto-build the engine from the ONNX on first run via `TRT::compile`, so no `trtexec` script is needed. Run these through the Makefile targets: the Makefile exports `LD_LIBRARY_PATH`, which TensorRT needs so its builder can `dlopen` `libnvinfer_builder_resource_smXX.so`. When running `./pro` directly, export `<lean_tensor_rt>/lib` first.

## Architecture

### Layer map

```
src/main.cpp                    — CLI dispatch
src/application/app_*.cpp       — One entry per task (app_yolo.cpp, app_rtmo.cpp, ...)
src/application/app_<task>/     — Model-specific headers, decode CUDA kernels, inference impl
src/tensorRT/
    builder/trt_builder.*       — Engine compilation: TRT::compile() for FP32/FP16/INT8
    infer/trt_infer.*           — Engine loading + runtime inference: TRT::load_infer()
    common/
        trt_tensor.*            — GPU tensor abstraction (TRT::Tensor, MixMemory)
        infer_controller.hpp    — Template base for async batch inference with worker thread
        preprocess_kernel.cu    — GPU preprocessing (resize, normalize, BGR→RGB)
        monopoly_allocator.hpp  — GPU memory pool allocator
        cuda_tools.*            — CUDA error checking, device info helpers
        ilogger.*               — Logging
    onnxplugin/                 — Custom ONNX plugins (LayerNorm, DCNv2, HSigmoid, HSwish, ScatterND)
```

### Application pattern

Every model task follows this structure:

1. **Header** (`app_<task>/<task>.hpp`) defines a namespace with:
   - `create_infer(engine_file, ...)` factory → `shared_ptr<Infer>`
   - `Infer` abstract class with `commit(image)` → `shared_future<Output>` and `commits(images)` → `vector<shared_future<Output>>`

2. **Implementation** (`app_<task>/<task>.cpp`) extends `InferController<cv::Mat, Output>`:
   - `preprocess()` — converts input image to GPU tensor
   - `worker()` — batch inference loop: wait for jobs → preprocess → TRT forward → postprocess/decode
   - Postprocessing often has a CUDA kernel (`app_<task>/<task>_decode.cu`) for GPU-parallel decode

3. **Entry point** (`app_<task>.cpp`) handles model compilation (on first run) and loads the inference engine.

### TRT 8.x / 10.x dual-version support

All API differences are handled via `#if NV_TENSORRT_MAJOR >= 10` preprocessor guards in `trt_builder.cpp` and `trt_infer.cpp`:

| Concern | TRT 8.x | TRT 10.x |
|---|---|---|
| Build serialization | `builder->buildEngineWithConfig()` + `engine->serialize()` | `builder->buildSerializedNetwork()` directly |
| Memory limit | `config->setMaxWorkspaceSize()` | `config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, ...)` |
| max batch size | `builder->setMaxBatchSize()` | Set via optimization profile dims only |
| Pointer destruction | `ptr->destroy()` | `delete ptr` |
| Removed enums | `kFULLY_CONNECTED`, `kRNN_V2`, `kCAFFE_ROUND_*` | Not available |
| Dims array access | `dims.d` (C-array) | Same, but used with explicit copy for safety |

### Defaults

- C++11, debug build (`-O0 -g`)
- Maximum workspace: 1 GB
- `EXPLICIT_BATCH` network creation flag
- Dynamic batch enabled via optimization profiles (min=1, max=compile-time config)

### Key dependencies

- NVIDIA TensorRT (8.x or 10.x) with `libnvonnxparser.so`
- CUDA + cuDNN
- OpenCV 4.x
- No protobuf dependency
