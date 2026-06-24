#include "rfdetr.hpp"
#include <common/cuda_tools.hpp>

namespace RFDETR{

    const int NUM_BOX_ELEMENT = 6;  // left, top, right, bottom, confidence, class

    static __global__ void decode_kernel(
        float *predict, int num_bboxes,
        float confidence_threshold,
        float* parray, int MAX_IMAGE_BOXES
    ){
        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= num_bboxes) return;

        // output format: [cx, cy, w, h, confidence, class_id]
        // class_id is 1-indexed COCO ID, convert to 0-indexed for cocolabels[]
        float* pitem  = predict + 6 * position;
        float cx      = pitem[0];
        float cy      = pitem[1];
        float width   = pitem[2];
        float height  = pitem[3];
        float confidence = pitem[4];
        int class_id  = (int)pitem[5];

        if(confidence < confidence_threshold)
            return;

        int index = atomicAdd(parray, 1);
        if(index >= MAX_IMAGE_BOXES)
            return;

        // convert center-format to corner-format
        float left   = cx - width  * 0.5f;
        float top    = cy - height * 0.5f;
        float right  = cx + width  * 0.5f;
        float bottom = cy + height * 0.5f;

        float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
        *pout_item++ = left;
        *pout_item++ = top;
        *pout_item++ = right;
        *pout_item++ = bottom;
        *pout_item++ = confidence;
        *pout_item++ = (float)class_id;  // raw COCO class ID (1-90)
    }

    void decode_kernel_invoker(
        float* predict, int num_bboxes,
        float confidence_threshold, float* parray,
        int max_objects, cudaStream_t stream
    ){
        auto grid  = CUDATools::grid_dims(num_bboxes);
        auto block = CUDATools::block_dims(num_bboxes);

        checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(
            predict, num_bboxes, confidence_threshold,
            parray, max_objects
        ));
    }
};
