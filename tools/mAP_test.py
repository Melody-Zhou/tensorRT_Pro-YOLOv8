import os
import copy
import logging
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging(__name__)

def eval_model(nc, names, anno_json, pred_json):
    anno = COCO(anno_json)
    pred = anno.loadRes(pred_json)
    cocoEval = COCOeval(anno, pred, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()

    val_dataset_img_count = cocoEval.cocoGt.imgToAnns.__len__()
    val_dataset_anns_count = 0
    label_count_dict = {"images":set(), "anns":0}
    label_count_dicts = [copy.deepcopy(label_count_dict) for _ in range(nc)]
    for _, ann_i in cocoEval.cocoGt.anns.items():
        if ann_i["ignore"]:
            continue
        val_dataset_anns_count += 1
        nc_i = ann_i['category_id']
        label_count_dicts[nc_i]["images"].add(ann_i["image_id"])
        label_count_dicts[nc_i]["anns"] += 1

    s = ('%-16s' + '%12s' * 7) % ('Class', 'Labeled_images', 'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
    LOGGER.info(s)
    #IOU , all p, all cats, all gt, maxdet 100
    coco_p = cocoEval.eval['precision']
    coco_p_all = coco_p[:, :, :, 0, 2]
    map = np.mean(coco_p_all[coco_p_all>-1])

    coco_p_iou50 = coco_p[0, :, :, 0, 2]
    map50 = np.mean(coco_p_iou50[coco_p_iou50>-1])
    mp = np.array([np.mean(coco_p_iou50[ii][coco_p_iou50[ii]>-1]) for ii in range(coco_p_iou50.shape[0])])
    mr = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
    mf1 = 2 * mp * mr / (mp + mr + 1e-16)
    i = mf1.argmax()  # max F1 index

    pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format
    LOGGER.info(pf % ('all', val_dataset_img_count, val_dataset_anns_count, mp[i], mr[i], mf1[i], map50, map))

    #compute each class best f1 and corresponding p and r
    for nc_i in range(nc):
        coco_p_c = coco_p[:, :, nc_i, 0, 2]
        map = np.mean(coco_p_c[coco_p_c>-1])

        coco_p_c_iou50 = coco_p[0, :, nc_i, 0, 2]
        map50 = np.mean(coco_p_c_iou50[coco_p_c_iou50>-1])
        p = coco_p_c_iou50
        r = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        f1 = 2 * p * r / (p + r + 1e-16)
        i = f1.argmax()
        LOGGER.info(pf % (names[nc_i], len(label_count_dicts[nc_i]["images"]), label_count_dicts[nc_i]["anns"], p[i], r[i], f1[i], map50, map))
    cocoEval.summarize()

if __name__ == "__main__":

    """
    1. mAP 测试程序, 会计算出每一类的 AP 值
    2. 需要指定类别数和对应的类别名称
    3. anno_json 为真实标签, pred_json 为预测标签
    """

    nc = 20
    names = ["aeroplane",   "bicycle", "bird",   "boat",       "bottle",
             "bus",         "car",     "cat",    "chair",      "cow",
             "diningtable", "dog",     "horse",  "motorbike",  "person",
             "pottedplant",  "sheep",  "sofa",   "train",      "tvmonitor"]
    anno_json = "workspace/val.json"
    pred_json = "workspace/best.minmax.INT8.trtmodel.prediction.json"
    eval_model(nc, names, anno_json, pred_json)
