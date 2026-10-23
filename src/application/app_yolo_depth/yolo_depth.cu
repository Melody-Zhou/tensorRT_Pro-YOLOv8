#include "yolo_depth.hpp"
#include <common/cuda_tools.hpp>

namespace YoloDepth{

    __global__ void resize_depth_align_corners_kernel(
        const float* src, int src_width, int src_height,
        float* dst, int dst_width, int dst_height, int edge
    ){
        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= edge) return;

        int dx = position % dst_width;
        int dy = position / dst_width;

        // align_corners=True coordinate mapping (same as torch F.interpolate)
        float scale_x = dst_width  > 1 ? (src_width  - 1) / (float)(dst_width  - 1) : 0.0f;
        float scale_y = dst_height > 1 ? (src_height - 1) / (float)(dst_height - 1) : 0.0f;
        float src_x   = dx * scale_x;
        float src_y   = dy * scale_y;

        int x_low  = min((int)src_x, src_width  - 1);
        int y_low  = min((int)src_y, src_height - 1);
        int x_high = min(x_low  + 1, src_width  - 1);
        int y_high = min(y_low  + 1, src_height - 1);

        float lx = src_x - x_low;
        float ly = src_y - y_low;
        float hx = 1.0f - lx;
        float hy = 1.0f - ly;

        float v1 = src[y_low  * src_width + x_low];
        float v2 = src[y_low  * src_width + x_high];
        float v3 = src[y_high * src_width + x_low];
        float v4 = src[y_high * src_width + x_high];

        dst[position] = hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
    }

    void resize_depth_align_corners_invoker(
        const float* src, int src_width, int src_height,
        float* dst, int dst_width, int dst_height,
        cudaStream_t stream
    ) {
        int jobs   = dst_width * dst_height;
        auto grid  = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);

        checkCudaKernel(resize_depth_align_corners_kernel<<<grid, block, 0, stream>>>(
            src, src_width, src_height,
            dst, dst_width, dst_height, jobs
        ));
    }
};
