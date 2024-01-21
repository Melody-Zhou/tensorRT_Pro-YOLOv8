import cv2
import torch
import numpy as np
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend

def preprocess_letterbox(image):
    letterbox = LetterBox(new_shape=1024, stride=32, auto=True)
    image = letterbox(image=image)
    image = (image[..., ::-1] / 255.0).astype(np.float32) # BGR to RGB, 0 - 255 to 0.0 - 1.0
    image = image.transpose(2, 0, 1)[None]  # BHWC to BCHW (n, 3, h, w)
    image = torch.from_numpy(image)
    return image

def preprocess_warpAffine(image, dst_width=1024, dst_height=1024):
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

def xywhr2xyxyxyxy(center):
    # reference: https://github.com/ultralytics/ultralytics/blob/v8.1.0/ultralytics/utils/ops.py#L545
    is_numpy = isinstance(center, np.ndarray)
    cos, sin = (np.cos, np.sin) if is_numpy else (torch.cos, torch.sin)

    ctr = center[..., :2]
    w, h, angle = (center[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1) if is_numpy else torch.cat(vec1, dim=-1)
    vec2 = np.concatenate(vec2, axis=-1) if is_numpy else torch.cat(vec2, dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=-2) if is_numpy else torch.stack([pt1, pt2, pt3, pt4], dim=-2)

def probiou(obb1, obb2, eps=1e-7):
    # Calculate the prob iou between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.
    def covariance_matrix(obb):
        # Extract elements
        w, h, r = obb[2:5]
        a = (w ** 2) / 12
        b = (h ** 2) / 12

        cos_r = torch.cos(torch.tensor(r))
        sin_r = torch.sin(torch.tensor(r))
        
        # Calculate covariance matrix elements
        a_val = a * cos_r ** 2 + b * sin_r ** 2
        b_val = a * sin_r ** 2 + b * cos_r ** 2
        c_val = (a - b) * sin_r * cos_r

        return a_val, b_val, c_val

    a1, b1, c1 = covariance_matrix(obb1)
    a2, b2, c2 = covariance_matrix(obb2)

    x1, y1 = obb1[:2]
    x2, y2 = obb2[:2]

    t1 = ((a1 + a2) * ((y1 - y2) ** 2) + (b1 + b2) * ((x1 - x2) ** 2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)
    t3 = torch.log(((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2) / (4 * torch.sqrt(a1 * b1 - c1 ** 2) * torch.sqrt(a2 * b2 - c2 ** 2) + eps) + eps)

    bd = 0.25 * t1 + 0.5 * t2 + 0.5 * t3
    hd = torch.sqrt(1.0 - torch.exp(-torch.clamp(bd, eps, 100.0)) + eps)
    return 1 - hd

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
            if(ibox[6] != jbox[6]):
                continue
            if probiou(ibox, jbox) > iou_thres:
                remove_flags[j] = True
    return keep_boxes

def postprocess(pred, IM=[], conf_thres=0.25, iou_thres=0.45):

    # 输入是模型推理的结果，即21504个预测框
    # 1,21504,20 [cx,cy,w,h,class*15,rotated]
    boxes = []
    for item in pred[0]:
        cx, cy, w, h = item[:4]
        angle = item[-1]
        label = item[4:-1].argmax()
        confidence = item[4 + label]
        if confidence < conf_thres:
            continue
        boxes.append([cx, cy, w, h, angle, confidence, label])

    boxes = np.array(boxes)
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    wh = boxes[:, 2:4]
    boxes[:, 0] = IM[0][0] * cx + IM[0][2]
    boxes[:, 1] = IM[1][1] * cy + IM[1][2]
    boxes[:, 2:4] = IM[0][0] * wh
    boxes = sorted(boxes.tolist(), key=lambda x:x[5], reverse=True)
    
    return NMS(boxes, iou_thres)

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

    img = cv2.imread("P0032.jpg")

    # img_pre = preprocess_letterbox(img)
    img_pre, IM = preprocess_warpAffine(img)
    model  = AutoBackend(weights="yolov8s-obb.pt")
    names  = model.names
    result = model(img_pre)[0].transpose(-1, -2)  # 1,21504,20

    boxes   = postprocess(result, IM)
    confs   = [box[5] for box in boxes]
    classes = [int(box[6]) for box in boxes]
    boxes   = xywhr2xyxyxyxy(np.array(boxes)[..., :5])

    for i, box in enumerate(boxes):
        confidence = confs[i]
        label = classes[i]
        color = random_color(label)
        cv2.polylines(img, [np.asarray(box, dtype=int)], True, color, 2)
        caption = f"{names[label]} {confidence:.2f}"
        w, h = cv2.getTextSize(caption, 0 ,1, 2)[0]
        left, top = [int(b) for b in box[0]]
        cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
        cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)
    
    cv2.imwrite("infer-obb.jpg", img)
    print("save done")