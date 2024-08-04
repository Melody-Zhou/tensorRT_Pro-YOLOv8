
#include "laneatt.hpp"
#include <common/cuda_tools.hpp>

namespace LaneATT{

    const int NUM_LANE_ELEMENT = 6 + N_OFFSETS;    // _, score, start_y, start_x, length, keepflag, (2*36)

    static __global__ void decode_kernel(float* predict, int num_lanes, float confidence_threshold, float* parray){

        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= num_lanes)  return;

        float* pitem = predict + (NUM_LANE_ELEMENT - 1) * position;
        float conf   = pitem[1];
        if(conf < confidence_threshold)
            return;
        
        int index = atomicAdd(parray, 1);
        float conf1   = *pitem++;
        float conf2   = *pitem++;
        float start_y = *pitem++;
        float start_x = *pitem++;
        float length  = *pitem++;

        float* pout_item = parray + 1 + index * NUM_LANE_ELEMENT;
        *pout_item++ = conf1;
        *pout_item++ = conf2;
        *pout_item++ = start_y;
        *pout_item++ = start_x;
        *pout_item++ = length;
        *pout_item++ = 1;   // 1 = keep, 0 = ignore

        for(int i = 0; i < N_OFFSETS; ++i){
            float point  = *pitem++;
            *pout_item++ = point;
        }
    }

    static __device__ float LaneIoU(float* a, float* b){
        int start_a = (int)(a[2] * N_STRIPS + 0.5f);
        int start_b = (int)(b[2] * N_STRIPS + 0.5f);
        int start   = max(start_a, start_b);
        int end_a   = start_a + (int)(a[4] + 0.5f) - 1;
        int end_b   = start_b + (int)(b[4] + 0.5f) - 1;
        int end     = min(min(end_a, end_b), N_STRIPS);
        float dist  = 0.0f;
        for(int i = 6 + start; i <= 6 + end; ++i){
            dist += fabsf(a[i] - b[i]);
        }
        return dist / (float)(end - start + 1);
    }

    static __global__ void nms_kernel(float* lanes, int max_lanes, float threshold){
        
        int position = (blockDim.x * blockIdx.x + threadIdx.x);
        int count = min((int)*lanes, max_lanes);
        if(position >= count)
            return;

        float* pcurrent = lanes + 1 + position * NUM_LANE_ELEMENT;
        if(pcurrent[5] == 0) return;

        for(int i = 0; i < count; ++i){
            float* pitem = lanes + 1 + i * NUM_LANE_ELEMENT;
            if(i == position)   continue;
            
            if(pitem[1] >= pcurrent[1]){
                if(pitem[1] == pcurrent[1] && i < position)
                    continue;
                
                float iou = LaneIoU(pcurrent, pitem);
                if(iou < threshold){
                    pcurrent[5] = 0;  // 1=keep, 0=ignore
                    return;
                }
            }
        }
    }

    void decode_kernel_invoker(float* predict, int num_lanes, float confidence_threshold, float* parray, cudaStream_t stream){
        
        auto grid  = CUDATools::grid_dims(num_lanes);
        auto block = CUDATools::block_dims(num_lanes);
        checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, num_lanes, confidence_threshold, parray));
    }

    void nms_kernel_invoker(float* parray, float nms_threshold, int max_lanes, cudaStream_t stream){
        
        auto grid  = CUDATools::grid_dims(max_lanes);
        auto block = CUDATools::block_dims(max_lanes);
        checkCudaKernel(nms_kernel<<<grid, block, 0, stream>>>(parray, max_lanes, nms_threshold));
    }
};