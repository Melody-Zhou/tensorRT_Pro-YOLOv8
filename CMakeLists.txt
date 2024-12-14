cmake_minimum_required(VERSION 2.6)
project(pro)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

# 如果你是不同显卡，请设置为显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
#set(CUDA_GEN_CODE "-gencode=arch=compute_75,code=sm_75")

# 如果你的opencv找不到，可以自己指定目录
set(OpenCV_DIR   "/usr/local/include/opencv4")

set(CUDA_TOOLKIT_ROOT_DIR     "/usr/local/cuda-11.6")
set(CUDNN_DIR    "/usr/local/cudnn8.4.0.27-cuda11.6")
# set(TENSORRT_DIR "/opt/TensorRT-8.4.1.5")
# RT-DETR 必须指定高版本的 tensorRT
set(TENSORRT_DIR "/home/jarvis/lean/TensorRT-8.6.1.6")

# 因为protobuf，需要用特定版本，所以这里指定路径
set(PROTOBUF_DIR "/home/jarvis/protobuf")

find_package(CUDA REQUIRED)
find_package(OpenCV)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/application
    ${PROJECT_SOURCE_DIR}/src/tensorRT
    ${PROJECT_SOURCE_DIR}/src/tensorRT/common
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${PROTOBUF_DIR}/include
    ${TENSORRT_DIR}/include
    ${CUDNN_DIR}/include
    ${OpenCV_DIR}/include/opencv4
)

# 切记，protobuf的lib目录一定要比tensorRT目录前面，因为tensorRTlib下带有protobuf的so文件
# 这可能带来错误
link_directories(
    ${PROTOBUF_DIR}/lib
    ${TENSORRT_DIR}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDNN_DIR}/lib
    ${OpenCV_DIR}/lib
)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")

option(BUILD_PANGO_BACKEND "Build with Pango backend" OFF)
option(BUILD_TRUETYPE_BACKEND "Build with TrueType backend" ON)

if(BUILD_PANGO_BACKEND)
    set(PANGO_LIBS ${PANGO_LIBS} pango-1.0 cairo pangocairo-1.0 glib-2.0 gobject-2.0)
    add_definitions(-DENABLE_TEXT_BACKEND_PANGO)
endif()

if(BUILD_TRUETYPE_BACKEND)
    add_definitions(-DENABLE_TEXT_BACKEND_STB)
endif()

file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu)
cuda_add_library(plugin_list SHARED ${cuda_srcs})
target_link_libraries(plugin_list nvinfer nvinfer_plugin)
target_link_libraries(plugin_list cuda cublas cudart cudnn)
target_link_libraries(plugin_list protobuf pthread)
target_link_libraries(plugin_list ${OpenCV_LIBS})
target_link_libraries(plugin_list opencv_core opencv_imgproc opencv_videoio opencv_highgui opencv_imgcodecs)

########################## custom_layernorm.so ################################
cuda_add_library(custom_layernorm SHARED
  src/tensorRT/onnxplugin/plugins/custom_layernorm.cu
)

target_link_libraries(custom_layernorm
  libnvinfer.so
  libnvinfer_plugin.so
)

add_executable(pro ${cpp_srcs})

# 如果提示插件找不到，请使用dlopen(xxx.so, NOW)的方式手动加载可以解决插件找不到问题
target_link_libraries(pro nvinfer nvinfer_plugin)
target_link_libraries(pro cuda cublas cudart cudnn)
target_link_libraries(pro protobuf pthread plugin_list)
target_link_libraries(pro ${OpenCV_LIBS})
target_link_libraries(pro opencv_core opencv_imgproc opencv_videoio opencv_highgui opencv_imgcodecs)
# target_link_libraries(pro dl)
# target_link_libraries(pro ${PANGO_LIBS})

add_custom_target(
    yolo
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro yolo
)

add_custom_target(
    yolo_pose
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro yolo_pose
)

add_custom_target(
    yolo_cls
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro yolo_cls
)

add_custom_target(
    yolo_seg
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro yolo_seg
)

add_custom_target(
    yolo_obb
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro yolo_obb
)

add_custom_target(
    bytetrack
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro bytetrack
)

add_custom_target(
    rtdetr
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro rtdetr
)

add_custom_target(
    rtmo
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro rtmo
)

add_custom_target(
    ppocr
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro ppocr
)

add_custom_target(
    laneatt
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro laneatt
)

add_custom_target(
    clrnet
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro clrnet
)

add_custom_target(
    clrernet
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro clrernet
)

add_custom_target(
    depth_anything
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro depth_anything
)

add_custom_target(
    test_yolo_map
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro test_yolo_map
)