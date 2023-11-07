import cv2
import numpy as np

def crop_resize(img, dst_width=224, dst_height=224):
    imh, imw = img.shape[:2]
    m = min(imh, imw)
    top, left = (imh - m) // 2, (imw - m) // 2
    img_crop = img[top:top+m, left:left+m]
    img_crop = cv2.resize(img_crop, (224, 224))
    return img_crop

def save_tensor(tensor, file):

    with open(file, "wb") as f:
        typeid = 0
        if tensor.dtype == np.float32:
            typeid = 0
        elif tensor.dtype == np.float16:
            typeid = 1
        elif tensor.dtype == np.int32:
            typeid = 2
        elif tensor.dtype == np.uint8:
            typeid = 3

        head = np.array([0xFCCFE2E2, tensor.ndim, typeid], dtype=np.uint32).tobytes()
        f.write(head)
        f.write(np.array(tensor.shape, dtype=np.uint32).tobytes())
        f.write(tensor.tobytes())

def load_tensor(file):
    
    with open(file, "rb") as f:
        binary_data = f.read()

    magic_number, ndims, dtype = np.frombuffer(binary_data, np.uint32, count=3, offset=0)
    assert magic_number == 0xFCCFE2E2, f"{file} not a tensor file."
    
    dims = np.frombuffer(binary_data, np.uint32, count=ndims, offset=3 * 4)

    if dtype == 0:
        np_dtype = np.float32
    elif dtype == 1:
        np_dtype = np.float16
    elif dtype == 2:
        np_dtype = np.int32
    elif dtype == 3:
        np_dtype = np.uint8
    else:
        assert False, f"Unsupport dtype = {dtype}, can not convert to numpy dtype"
        
    return np.frombuffer(binary_data, np_dtype, offset=(ndims + 3) * 4).reshape(*dims)

if __name__ == "__main__":

    # test crop resize
    img = cv2.imread("workspace/inference/car.jpg")
    img_crop = crop_resize(img)
    # cv2.imwrite("crop_resize_python.jpg", img_crop)
    # save_tensor(img_crop, "workspace/crop_resize_python.bin")
    crop_resize_python = load_tensor("workspace/crop_resize_python.bin")

    # 1, 3, 224, 224 => b, c, h, w
    crop_resize_cuda   = load_tensor("workspace/crop_resize_cuda.bin")[0].transpose(1, 2, 0)
    # cv2.imwrite("crop_resize_cuda.jpg", crop_resize_cuda)

    print("===========================Python Crop Resize==============================")
    print(crop_resize_python)
    
    print("\n\n")
    print("===========================CUDA Crop Resize==============================")
    print(crop_resize_cuda)

    print("\n\n")
    print("===========================abs diff crop resize==============================")
    abs_diff_python_cuda = np.abs(crop_resize_python.astype(int) - crop_resize_cuda.astype(int)).sum()
    print(abs_diff_python_cuda)