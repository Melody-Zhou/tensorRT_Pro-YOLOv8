import cv2
import torch
import numpy as np
import torch.nn.functional as F
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend

def preprocess_letterbox(image):
    letterbox = LetterBox(new_shape=640, stride=32, auto=True)
    image = letterbox(image=image)
    image = (image[..., ::-1] / 255.0).astype(np.float32) # BGR to RGB, 0 - 255 to 0.0 - 1.0
    image = image.transpose(2, 0, 1)[None]  # BHWC to BCHW (n, 3, h, w)
    image = torch.from_numpy(image)
    return image

def preprocess_warpAffine(image, dst_width=640, dst_height=640):
    scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
    ox = (dst_width  - scale * image.shape[1]) / 2
    oy = (dst_height - scale * image.shape[0]) / 2
    M = np.array([
        [scale, 0, ox],
        [0, scale, oy]
    ], dtype=np.float32)
    
    img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    IM = cv2.invertAffineTransform(M)

    img_pre = (img_pre[...,::-1] / 255.0).astype(np.float32)
    img_pre = img_pre.transpose(2, 0, 1)[None]
    img_pre = torch.from_numpy(img_pre)
    return img_pre, IM

def iou(box1, box2):
    def area_box(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    
    left, top = max(box1[:2], box2[:2])
    right, bottom = min(box1[2:4], box2[2:4])
    union = max((right-left), 0) * max((bottom-top), 0)
    cross = area_box(box1) + area_box(box2) - union
    if cross == 0 or union == 0:
        return 0
    return union / cross

def NMS(boxes, iou_thres):
    
    remove_flags = [False] * len(boxes)

    keep_boxes = []
    for i, ibox in enumerate(boxes):
        if remove_flags[i]:
            continue

        keep_boxes.append(ibox)
        for j in range(i + 1, len(boxes)):
            if remove_flags[j]:
                continue

            jbox = boxes[j]
            if(ibox[5] != jbox[5]):
                continue
            if iou(ibox, jbox) > iou_thres:
                remove_flags[j] = True
    return keep_boxes

def postprocess(pred, conf_thres=0.25, iou_thres=0.45):

    # 输入是模型推理的结果，即8400个预测框
    # 1,8400,116 [cx,cy,w,h,class*80,32]
    boxes = []
    for item in pred[0]:
        cx, cy, w, h = item[:4]
        label = item[4:-32].argmax()
        confidence = item[4 + label]
        if confidence < conf_thres:
            continue
        left    = cx - w * 0.5
        top     = cy - h * 0.5
        right   = cx + w * 0.5
        bottom  = cy + h * 0.5
        boxes.append([left, top, right, bottom, confidence, label, *item[-32:]])
        
    boxes = sorted(boxes, key=lambda x:x[4], reverse=True)

    return NMS(boxes, iou_thres)

def crop_mask(masks, boxes):
    
    # masks -> n, 160, 160  原始 masks
    # boxes -> n, 4         检测框，映射到 160x160 尺寸下的
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def process_mask(protos, masks_in, bboxes, shape, upsample=False):

    # protos   -> 32, 160, 160 分割头输出
    # masks_in -> n, 32        检测头输出的 32 维向量，可以理解为 mask 的权重
    # bboxes   -> n, 4         检测框
    # shape    -> 640, 640     输入网络中的图像 shape
    # unsample 一个 bool 值，表示是否需要上采样 masks 到图像的原始形状
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    # 矩阵相乘 nx32 @ 32x(160x160) -> nx(160x160) -> sigmoid -> nx160x160
    masks = (masks_in.float() @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    return masks.gt_(0.5)

def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)

def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)

if __name__ == "__main__":

    img = cv2.imread("ultralytics/assets/bus.jpg")

    # img_pre = preprocess_letterbox(img)
    img_pre, IM = preprocess_warpAffine(img)

    model  = AutoBackend(weights="yolov8s-seg.pt")
    names  = model.names
    result = model(img_pre)
    """
    result[0] -> 1, 116, 8400 -> det head
    result[1][0][0] -> 1, 144, 80, 80
    result[1][0][1] -> 1, 144, 40, 40
    result[1][0][2] -> 1, 144, 20, 20
    result[1][1] -> 1, 32, 8400
    result[1][2] -> 1, 32, 160, 160 -> seg head
    """

    output0 = result[0].transpose(-1, -2) # 1,8400,116 检测头输出
    output1 = result[1][2][0]             # 32,160,160 分割头输出

    pred = postprocess(output0)
    pred = torch.from_numpy(np.array(pred).reshape(-1, 38))

    # pred -> nx38 = [cx,cy,w,h,conf,label,32]
    masks = process_mask(output1, pred[:, 6:], pred[:, :4], img_pre.shape[2:], True)

    boxes = np.array(pred[:,:6])
    lr = boxes[:, [0, 2]]
    tb = boxes[:,[1, 3]]
    boxes[:,[0, 2]] = IM[0][0] * lr + IM[0][2]
    boxes[:,[1, 3]] = IM[1][1] * tb + IM[1][2]

    # draw mask
    h, w = img.shape[:2]
    for i, mask in enumerate(masks):
        
        mask = mask.cpu().numpy().astype(np.uint8) # 640x640
        mask_resized = cv2.warpAffine(mask, IM, (w, h), flags=cv2.INTER_LINEAR)  # 1080x810
        
        label = int(boxes[i][5])
        color = np.array(random_color(label))
        
        colored_mask = (np.ones((h, w, 3)) * color).astype(np.uint8)
        masked_colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_resized)

        mask_indices = mask_resized == 1
        img[mask_indices] = (img[mask_indices] * 0.6 + masked_colored_mask[mask_indices] * 0.4).astype(np.uint8)

        # contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img, contours, -1, random_color(label), 2)

    # draw box
    for obj in boxes:
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]
        label = int(obj[5])
        color = random_color(label)
        cv2.rectangle(img, (left, top), (right, bottom), color = color ,thickness=2, lineType=cv2.LINE_AA)
        caption = f"{names[label]} {confidence:.2f}"
        w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
        cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
        cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)
    
    cv2.imwrite("infer-seg.jpg", img)
    print("save done")    