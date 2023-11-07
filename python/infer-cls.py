import cv2
import torch
import numpy as np
from ultralytics.nn.autobackend import AutoBackend

def preprocess(img, dst_width=224, dst_height=224):

    imh, imw = img.shape[:2]
    m = min(imh, imw)
    top, left = (imh - m) // 2, (imw - m) // 2
    img_pre = img[top:top+m, left:left+m]
    img_pre = cv2.resize(img_pre, (dst_width, dst_height), interpolation=cv2.INTER_LINEAR)
    
    img_pre = (img_pre[...,::-1] / 255.0).astype(np.float32)
    img_pre = img_pre.transpose(2, 0, 1)[None]
    img_pre = torch.from_numpy(img_pre)

    return img_pre

if __name__ == "__main__":

    img = cv2.imread("ultralytics/assets/bus.jpg")

    img_pre = preprocess(img)

    model = AutoBackend(weights="yolov8s-cls.pt")
    names = model.names
    probs = model(img_pre)[0]

    top1_label = int(probs.argmax())
    top5_label = (-probs).argsort(0)[:5].tolist()
    top1_conf  = probs[top1_label]
    top5_conf  = probs[top5_label]

    top1name = names[top1_label]

    print(f"The model predicted category is {top1name}, label = {top1_label}, confidence = {top1_conf:.4f}")