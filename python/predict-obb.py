import cv2
import torch
import numpy as np
from ultralytics import YOLO

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

    model = YOLO("yolov8s-obb.pt")

    img = cv2.imread("P0032.png")
    results = model(img)[0]
    names   = results.names
    boxes   = results.obb.data.cpu()
    confs   = boxes[..., 5].tolist()
    classes = list(map(int, boxes[..., 6].tolist()))
    boxes   = xywhr2xyxyxyxy(boxes[..., :5])
    
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

    cv2.imwrite("predict-obb.jpg", img)
    print("save done")    