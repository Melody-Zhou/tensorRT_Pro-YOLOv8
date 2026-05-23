#include "yolo_sem.hpp"
#include <common/cuda_tools.hpp>

namespace YoloSEM{

    template<typename _T>
    static __inline__ __device__ _T limit(_T value, _T low, _T high){
        return value < low ? low : (value > high ? high : value);
    }

    __global__ void warp_affine_semantic_mask_kernel(
        float* src, int src_width, int src_height, int num_classes,
        uint8_t* dst, int dst_width, int dst_height,
        float* matrix_2_3, int edge
    ){
        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= edge) return;

        float m_x1 = matrix_2_3[0];
        float m_y1 = matrix_2_3[1];
        float m_z1 = matrix_2_3[2];
        float m_x2 = matrix_2_3[3];
        float m_y2 = matrix_2_3[4];
        float m_z2 = matrix_2_3[5];

        int dx      = position % dst_width;
        int dy      = position / dst_width;
        float src_x = m_x1 * dx + m_y1 * dy + m_z1;
        float src_y = m_x2 * dx + m_y2 * dy + m_z2;

        if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
            // out of range, set ignore index
            dst[position] = 255;
            return;
        }

        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = limit(y_low + 1, 0, src_height - 1);
        int x_high = limit(x_low + 1, 0, src_width - 1);
        y_low = limit(y_low, 0, src_height - 1);
        x_low = limit(x_low, 0, src_width - 1);

        float ly    = src_y - y_low;
        float lx    = src_x - x_low;
        float hy    = 1 - ly;
        float hx    = 1 - lx;
        float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        int src_area = src_width * src_height;

        int idx_v1 = y_low  * src_width + x_low;
        int idx_v2 = y_low  * src_width + x_high;
        int idx_v3 = y_high * src_width + x_low;
        int idx_v4 = y_high * src_width + x_high;

        float max_val = -1e30f;
        int max_class = 0;
        for(int c = 0; c < num_classes; ++c){
            int base = c * src_area;
            float v1 = src[base + idx_v1];
            float v2 = src[base + idx_v2];
            float v3 = src[base + idx_v3];
            float v4 = src[base + idx_v4];
            float val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
            if(val > max_val){
                max_val = val;
                max_class = c;
            }
        }

        dst[position] = (uint8_t)max_class;
    }

    void warp_affine_semantic_mask_invoker(
        float* src, int src_width, int src_height, int num_classes,
        uint8_t* dst, int dst_width, int dst_height,
        float* matrix_2_3, cudaStream_t stream
    ) {
        int jobs   = dst_width * dst_height;
        auto grid  = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);

        checkCudaKernel(warp_affine_semantic_mask_kernel<<<grid, block, 0, stream>>>(
            src, src_width, src_height, num_classes,
            dst, dst_width, dst_height, matrix_2_3, jobs
        ));
    }
};
