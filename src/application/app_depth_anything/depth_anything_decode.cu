#include "depth_anything.hpp"
#include <common/cuda_tools.hpp>

namespace DepthAnything{

    #define INTER_RESIZE_COEF_BITS 11
    #define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)

    template<typename _T>
    static __inline__ __device__ _T limit(_T value, _T low, _T high){
        return value < low ? low : (value > high ? high : value);
    }

    __global__ void resize_bilinear_depth_kernel(
        float* src, int src_width, int src_height, 
        float* dst, int dst_width, int dst_height,
        float sx, float sy, int edge
    ){
        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= edge) return;

        int dx      = position % dst_width;
        int dy      = position / dst_width;
        float src_x = (dx + 0.5f) * sx - 0.5f;
        float src_y = (dy + 0.5f) * sy - 0.5f;

        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = limit(y_low + 1, 0, src_height - 1);
        int x_high = limit(x_low + 1, 0, src_width - 1);
        y_low = limit(y_low, 0, src_height - 1);
        x_low = limit(x_low, 0, src_width - 1);

        int ly = rint((src_y - y_low) * INTER_RESIZE_COEF_SCALE);
        int lx = rint((src_x - x_low) * INTER_RESIZE_COEF_SCALE);
        int hy = INTER_RESIZE_COEF_SCALE - ly;
        int hx = INTER_RESIZE_COEF_SCALE - lx;
        int w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        float v1 = src[y_low * src_width + x_low];
        float v2 = src[y_low * src_width + x_high];
        float v3 = src[y_high * src_width + x_low];
        float v4 = src[y_high * src_width + x_high];

        float interpolated_value = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4) / INTER_RESIZE_COEF_SCALE;
        dst[dy * dst_width + dx] = interpolated_value;
    }

    void resize_depth_image(
        float* src_depth, int src_width, int src_height,
        float* dst_depth, int dst_width, int dst_height,
        cudaStream_t stream
    ) {

        int jobs = dst_width * dst_height;
        auto grid = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);

        resize_bilinear_depth_kernel<<<grid, block, 0, stream>>>(
            src_depth, src_width, src_height, dst_depth, dst_width, dst_height,
            src_width/(float)dst_width, src_height/(float)dst_height, jobs
        );
    }
};