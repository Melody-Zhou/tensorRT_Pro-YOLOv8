import os
import cv2
import json
import logging
import os.path as osp
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging(__name__)

def process_img(image_filename, data_path, label_path):
    # Open the image file to get its size
    image_path = os.path.join(data_path, image_filename)
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Open the corresponding label file
    label_file = os.path.join(label_path, os.path.splitext(image_filename)[0] + ".txt")
    with open(label_file, "r") as file:
        lines = file.readlines()

    # Process the labels
    labels = []
    for line in lines:
        category, x, y, w, h = map(float, line.strip().split())
        labels.append((category, x, y, w, h))

    return image_filename, {"shape": (height, width), "labels": labels}

def get_img_info(data_path, label_path):
    LOGGER.info(f"Get img info")

    image_filenames = os.listdir(data_path)

    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(partial(process_img, data_path=data_path, label_path=label_path), image_filenames), total=len(image_filenames)))

    img_info = {image_filename: info for image_filename, info in results}
    return img_info


def generate_coco_format_labels(img_info, class_names, save_path):
    # for evaluation with pycocotools
    dataset = {"categories": [], "annotations": [], "images": []}
    for i, class_name in enumerate(class_names):
        dataset["categories"].append(
            {"id": i, "name": class_name, "supercategory": ""}
        )

    ann_id = 0
    LOGGER.info(f"Convert to COCO format")
    for i, (img_path, info) in enumerate(tqdm(img_info.items())):
        labels = info["labels"] if info["labels"] else []
        img_id = osp.splitext(osp.basename(img_path))[0]
        img_h, img_w = info["shape"]
        dataset["images"].append(
            {
                "file_name": os.path.basename(img_path),
                "id": img_id,
                "width": img_w,
                "height": img_h,
            }
        )
        if labels:
            for label in labels:
                c, x, y, w, h = label[:5]
                # convert x,y,w,h to x1,y1,x2,y2
                x1 = (x - w / 2) * img_w
                y1 = (y - h / 2) * img_h
                x2 = (x + w / 2) * img_w
                y2 = (y + h / 2) * img_h
                # cls_id starts from 0
                cls_id = int(c)
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                dataset["annotations"].append(
                    {
                        "area": h * w,
                        "bbox": [x1, y1, w, h],
                        "category_id": cls_id,
                        "id": ann_id,
                        "image_id": img_id,
                        "iscrowd": 0,
                        # mask
                        "segmentation": [],
                    }
                )
                ann_id += 1

    with open(save_path, "w") as f:
        json.dump(dataset, f)
        LOGGER.info(
            f"Convert to COCO format finished. Resutls saved in {save_path}"
        )


if __name__ == "__main__":
    
    """
    1. 该脚本文件用于将 yolo 格式的标签转换成 json 文件, 用于 mAP 的计算
    2. 生成的 json 文件保存在 workspace/val.json
    """
    
    # Define the paths
    data_path   = "/home/jarvis/Learn/Datasets/VOC_PTQ/images/val"
    label_path  = "/home/jarvis/Learn/Datasets/VOC_PTQ/labels/val"

    # 类别名称请务必与 YOLO 格式的标签对应
    class_names = ["aeroplane",   "bicycle", "bird",   "boat",       "bottle",
                   "bus",         "car",     "cat",    "chair",      "cow",
                   "diningtable", "dog",     "horse",  "motorbike",  "person",
                   "pottedplant",  "sheep",  "sofa",   "train",      "tvmonitor"]
    save_path   = "workspace/val.json"

    img_info = get_img_info(data_path, label_path)
    generate_coco_format_labels(img_info, class_names, save_path)