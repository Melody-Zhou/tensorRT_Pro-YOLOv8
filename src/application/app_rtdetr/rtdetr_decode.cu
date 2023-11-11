
#include "rtdetr.hpp"
#include <common/cuda_tools.hpp>

namespace RTDETR{

    const int NUM_BOX_ELEMENT = 6;      // left, top, right, bottom, confidence, class

    static __global__ void decode_kernel(float *predict, int num_bboxes, int num_classes, float confidence_threshold, float* parray, int MAX_IMAGE_BOXES){
        
        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= num_bboxes) return;

        float* pitem            = predict + (4 + num_classes) * position;
        float* class_confidence = pitem + 4;
        float confidence        = *class_confidence++;
        int label               = 0;
        for(int i = 1; i < num_classes; ++i, ++class_confidence){
            if(*class_confidence > confidence){
                confidence = *class_confidence;
                label      = i;
            }
        }

        if(confidence < confidence_threshold)
            return;

        int index = atomicAdd(parray, 1);
        if(index >= MAX_IMAGE_BOXES)
            return;

        float cx         = *pitem++;
        float cy         = *pitem++;
        float width      = *pitem++;
        float height     = *pitem++;
        float left   = cx - width  * 0.5f;
        float top    = cy - height * 0.5f;
        float right  = cx + width  * 0.5f;
        float bottom = cy + height * 0.5f;

        float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
        *pout_item++ = left;
        *pout_item++ = top;
        *pout_item++ = right;
        *pout_item++ = bottom;
        *pout_item++ = confidence;
        *pout_item++ = label;
    }

    void decode_kernel_invoker(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* parray, int max_objects, cudaStream_t stream){
        
        auto grid = CUDATools::grid_dims(num_bboxes);
        auto block = CUDATools::block_dims(num_bboxes);
        checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, num_classes, confidence_threshold, parray, max_objects));
    }
};