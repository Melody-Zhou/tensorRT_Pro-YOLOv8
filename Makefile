cc        := g++
nvcc      = ${lean_cuda}/bin/nvcc

build_pango_backend    := no
build_truetype_backend := yes

lean_protobuf  := /home/jarvis/protobuf
# lean_tensor_rt := /opt/TensorRT-8.4.1.5
# RT-DETR 必须指定高版本的 tensorRT
lean_tensor_rt := /home/jarvis/lean/TensorRT-8.6.1.6
lean_cudnn     := /usr/local/cuda-11.6
lean_opencv    := /usr/local
lean_cuda      := /usr/local/cuda-11.6

# 如果是其他显卡，请修改-gencode=arch=compute_75,code=sm_75为对应显卡的能力
# 显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
cuda_arch := # -gencode=arch=compute_75,code=sm_75

cpp_srcs  := $(shell find src -name "*.cpp")
cpp_objs  := $(cpp_srcs:.cpp=.cpp.o)
cpp_objs  := $(cpp_objs:src/%=objs/%)
cpp_mk    := $(cpp_objs:.cpp.o=.cpp.mk)

cu_srcs  := $(shell find src -name "*.cu")
cu_objs  := $(cu_srcs:.cu=.cu.o)
cu_objs  := $(cu_objs:src/%=objs/%)
cu_mk    := $(cu_objs:.cu.o=.cu.mk)

include_paths := src        \
			src/application \
			src/tensorRT	\
			src/tensorRT/common  \
			$(lean_protobuf)/include \
			$(lean_opencv)/include/opencv4 \
			$(lean_tensor_rt)/include \
			$(lean_cuda)/include  \
			$(lean_cudnn)/include 

library_paths := $(lean_protobuf)/lib \
			$(lean_opencv)/lib    \
			$(lean_tensor_rt)/lib \
			$(lean_cuda)/lib64  \
			$(lean_cudnn)/lib

link_librarys := opencv_core opencv_imgproc opencv_videoio opencv_highgui opencv_imgcodecs \
			nvinfer nvinfer_plugin \
			cuda cublas cudart cudnn \
			stdc++ protobuf dl

ifeq ($(build_pango_backend), yes)
link_librarys += pango-1.0 cairo pangocairo-1.0 glib-2.0 gobject-2.0
cppdefine += -DENABLE_TEXT_BACKEND_PANGO
endif

ifeq ($(build_truetype_backend), yes)
cppdefine += -DENABLE_TEXT_BACKEND_STB
endif

empty         :=
export_path   := $(subst $(empty) $(empty),:,$(library_paths))

run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

cpp_compile_flags := -std=c++11 -g -w -O0 -fPIC -pthread -fopenmp $(cppdefine)
cu_compile_flags  := -std=c++11 -g -w -O0 -Xcompiler "$(cpp_compile_flags)" $(cuda_arch)
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        += $(library_paths) $(link_librarys) $(run_paths)

ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mk) $(cu_mk)
endif

pro    : workspace/pro
expath : library_path.txt

library_path.txt : 
	@echo LD_LIBRARY_PATH=$(export_path):"$$"LD_LIBRARY_PATH > $@

workspace/pro : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) $^ -o $@ $(link_flags)

objs/%.cpp.o : src/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

objs/%.cu.o : src/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

objs/%.cpp.mk : src/%.cpp
	@echo Compile depends CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)
	
objs/%.cu.mk : src/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)

yolo : workspace/pro
	@cd workspace && ./pro yolo

yolo_pose : workspace/pro
	@cd workspace && ./pro yolo_pose

yolo_cls : workspace/pro
	@cd workspace && ./pro yolo_cls

yolo_seg : workspace/pro
	@cd workspace && ./pro yolo_seg

yolo_obb : workspace/pro
	@cd workspace && ./pro yolo_obb

bytetrack : workspace/pro
	@cd workspace && ./pro bytetrack

rtdetr : workspace/pro
	@cd workspace && ./pro rtdetr

rtmo : workspace/pro
	@cd workspace && ./pro rtmo

ppocr : workspace/pro
	@cd workspace && ./pro ppocr

laneatt : workspace/pro
	@cd workspace && ./pro laneatt

clrnet : workspace/pro
	@cd workspace && ./pro clrnet

test_yolo_map : workspace/pro
	@cd workspace && ./pro test_yolo_map

clean :
	@rm -rf objs workspace/pro
	@rm -rf workspace/single_inference
	@rm -rf build
	@rm -rf library_path.txt

.PHONY : clean yolo  debug

# 导出符号，使得运行时能够链接上
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jarvis/lean/TensorRT-8.6.1.6/lib
export LD_LIBRARY_PATH:=$(export_path):$(LD_LIBRARY_PATH)