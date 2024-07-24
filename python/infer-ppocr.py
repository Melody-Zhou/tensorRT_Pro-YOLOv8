import cv2
import math
import copy
import random
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import pyclipper
from shapely.geometry import Polygon

class TextDetector(object):
    def __init__(self, model_path, mask_thresh=0.3, box_thresh=0.6, 
                 max_candidates=1000, min_size=3, unclip_ratio=1.5) -> None:
        self.predictor      = ort.InferenceSession(model_path, provider_options=["CPUExecutionProvider"])
        self.mask_thresh    = mask_thresh
        self.box_thresh     = box_thresh
        self.max_candidates = max_candidates
        self.min_size       = min_size
        self.unclip_ratio   = unclip_ratio

    def preprocess(self, img, tar_w=960, tar_h=960):
        # 1. resize
        img  = cv2.resize(img, (int(tar_w), int(tar_h)))
        # 2. normalize
        img  = img.astype("float32") / 255.0
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        mean = np.array(mean).reshape(1, 1, 3).astype("float32")
        std  = np.array(std).reshape(1, 1, 3).astype("float32")
        img  = (img - mean) / std
        # 3. to bchw
        img  = img.transpose((2, 0, 1))[None]
        return img
    
    def forward(self, input):
        # input->1x3x960x960
        output = self.predictor.run(None, {"images": input})[0]
        return output
    
    def postprocess(self, pred, src_h, src_w):
        # pred->1x1x960x960
        pred = pred[0, 0, :, :]
        mask = pred > self.mask_thresh
        boxes, _ = self._boxes_from_bitmap(pred, mask, src_w, src_h)
        boxes = self._filter_boxes(boxes, src_h, src_w)
        boxes = self._sorted_boxes(boxes)
        return boxes

    def _boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        """
        bitmap: single map with shape (H, W),
                whose values are binarized as {0, 1}
        """
        
        height, width = bitmap.shape
        
        # bitmap_image = (bitmap * 255).astype(np.uint8)
        # cv2.imwrite("bitmap_image.jpg", bitmap_image)

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            _, contours, _ = outs[0], outs[1], outs[2]  # opencv3.x
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]  # opencv4.x
        
        num_contours = min(len(contours), self.max_candidates)
        
        # contour_image = cv2.cvtColor(bitmap_image, cv2.COLOR_GRAY2BGR)
        # for contour in contours:
        #     cv2.drawContours(contour_image, [contour], -1, (0, 0, 255), 2)
        # cv2.imwrite('contour_image.jpg', contour_image)

        boxes  = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self._get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score  = self._box_score(pred, points.reshape(-1, 2))
            if score < self.box_thresh:
                continue
            
            box = self._unclip(points, self.unclip_ratio)
            if len(box) > 1:
                continue
            box = np.array(box).reshape(-1, 1, 2)
            box, sside = self._get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(np.round(box[:, 0] / width  * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    def _get_mini_boxes(self, contour):
        # [[center_x, center_y], [width, height], angle]
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def _box_score(self, bitmap, _box):
        """
        box_score: use bbox mean score as the mean score
        """
        h, w = bitmap.shape[:2]
        box  = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def _unclip(self, box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        return expanded

    def _filter_boxes(self, boxes, src_h, src_w):
        boxes_filter = []
        for box in boxes:
            box = self._order_points_clockwise(box)
            box = self._clip(box, src_h, src_w)
            rect_width  = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            boxes_filter.append(box)
        return np.array(boxes_filter)
            
    def _order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp  = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def _clip(self, points, img_height, img_width):
        for idx in range(points.shape[0]):
            points[idx, 0] = int(min(max(points[idx, 0], 0), img_width - 1))
            points[idx, 1] = int(min(max(points[idx, 1], 0), img_height - 1))
        return points

    def _sorted_boxes(self, boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        """
        num_boxes = boxes.shape[0]
        boxes_sorted = sorted(boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(boxes_sorted)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

class TextClassifier(object):
    def __init__(self, model_path, cls_thresh=0.9, cls_batch_num=6) -> None:
        self.predictor     = ort.InferenceSession(model_path, provider_options=["CPUExecutionProvider"])
        self.cls_thresh    = cls_thresh
        self.cls_batch_num = cls_batch_num
    
    def preprocess(self, img, boxes, tar_w=192, tar_h=48):
        img_crop_list = []
        for box in boxes:
            tmp_box  = copy.deepcopy(box)
            img_crop = self._get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)
        img_num = len(img_crop_list)
        ratio_list = [img.shape[1] / float(img.shape[0]) for img in img_crop_list]
        indices = np.argsort(np.array(ratio_list))

        imgs_pre_batch = []
        for beg_img_idx in range(0, img_num, self.cls_batch_num):
            end_img_idx = min(img_num, beg_img_idx + self.cls_batch_num)
            norm_img_batch = []
            for idx in range(beg_img_idx, end_img_idx):
                norm_img = self._resize_norm_img(img_crop_list[indices[idx]], tar_w, tar_h)
                norm_img = norm_img[None]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            imgs_pre_batch.append(norm_img_batch)

        return img_crop_list, imgs_pre_batch, indices

    def forward(self, inputs):
        # inputs->bx3x48x196
        output = self.predictor.run(None, {"images": inputs})[0]
        return output

    def postprocess(self, img_list, imgs_pre_batch, indices):
        cls_res = [["", 0.0]] * len(img_list)
        for batch in range(len(imgs_pre_batch)):
            # infer
            cls_pred = self.forward(imgs_pre_batch[batch])
            # cls_pred->bx2
            pred_idxs = cls_pred.argmax(axis=1)
            label_list = ["0", "180"]
            cls_result = [(label_list[idx], cls_pred[i, idx]) for i, idx in enumerate(pred_idxs)]
            for i in range(len(cls_result)):
                label, score = cls_result[i]
                cls_res[indices[batch * self.cls_batch_num + i]] = [label, score]
                if "180" in label and score > self.cls_thresh:
                    img_list[indices[batch * self.cls_batch_num + i]] = cv2.rotate(
                        img_list[indices[batch * self.cls_batch_num + i]], 1)
        return img_list, cls_res
    
    def _get_rotate_crop_image(self, img, points):
        img_crop_width  = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height]
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC
        )
        dst_img_height, dst_img_width = dst_img.shape[:2]
        if (dst_img_height * 1.0 / dst_img_width) >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def _resize_norm_img(self, img, dest_width, dest_height):
        h, w, _ = img.shape
        ratio = w / float(h)
        if math.ceil(dest_height * ratio) > dest_width:
            resized_w = dest_width
        else:
            resized_w = int(math.ceil(dest_height * ratio))
        resized_image = cv2.resize(img, (resized_w, dest_height))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((3, dest_height, dest_width), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

class TextRecognizer(object):
    def __init__(self, model_path, character_dict_path, rec_batch_num=6) -> None:
        self.predictor     = ort.InferenceSession(model_path, provider_options=["CPUExecutionProvider"])
        self.character_str = []
        self.rec_batch_num = rec_batch_num

        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode("utf-8").strip("\n").strip("\r\n")
                self.character_str.append(line)
        self.character_str.append(" ")
        self.character_str = ["blank"] + self.character_str
        self.dict = {}
        for i, char in enumerate(self.character_str):
            self.dict[char] = i
    
    def preprocess(self, img_list, tar_w=640, tar_h=48):
        # for img in img_list:
        #     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #     plt.show()
        img_num = len(img_list)
        ratio_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.array(ratio_list))
        
        imgs_pre_batch = []
        for beg_img_idx in range(0, img_num, self.rec_batch_num):
            end_img_idx = min(img_num, beg_img_idx + self.rec_batch_num)
            norm_img_batch = []
            for idx in range(beg_img_idx, end_img_idx):
                norm_img = self._resize_norm_img(img_list[indices[idx]], tar_w, tar_h)

                # processed_img = norm_img.transpose(1, 2, 0)
                # processed_img = (processed_img * 0.5 + 0.5) * 255
                # processed_img = processed_img.astype(np.uint8)
                # plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                # plt.show()

                norm_img = norm_img[None]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            imgs_pre_batch.append(norm_img_batch)
        
        return imgs_pre_batch, indices

    def forward(self, inputs):
        # inputs->bx3x48x640
        output = self.predictor.run(None, {"images": inputs})[0]
        return output        

    def postprocess(self, imgs_pre_batch, indices):
        rec_res = [["", 0.0]] * len(indices)
        for batch in range(len(imgs_pre_batch)):
            # infer
            rec_pred = self.forward(imgs_pre_batch[batch])
            # rec_pred->bx80x6625
            preds_idx  = rec_pred.argmax(axis=2)
            preds_prob = rec_pred.max(axis=2)
            text = self._decode(preds_idx, preds_prob)
            for i in range(len(text)):
                rec_res[indices[batch * self.rec_batch_num + i]] = text[i]
        return rec_res

    def _resize_norm_img(self, img, dest_width, dest_height):
        h, w, _ = img.shape
        ratio = w / float(h)
        if math.ceil(dest_height * ratio) > dest_width:
            resized_w = dest_width
        else:
            resized_w = int(math.ceil(dest_height * ratio))
        resized_image = cv2.resize(img, (resized_w, dest_height))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((3, dest_height, dest_width), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def _decode(self, text_index, text_prob):
        "convert text-index into text-label"
        result_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[0]), dtype=bool)
            selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            selection &= text_index[batch_idx] != 0
            char_list = [self.character_str[text_id] for text_id in text_index[batch_idx][selection]]
            conf_list = text_prob[batch_idx][selection]
            if len(conf_list) == 0:
                conf_list = 0
            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

def create_font(txt, sz, font_path):
    font_size = int(sz[1] * 0.99)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    length = font.getlength(txt)
    if(length > sz[0]):
        font_size = int(font_size * sz[0] / length)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font

def draw_box_txt(img_size, box, txt, font_path=None):
    box_height = int(np.linalg.norm(box[0] - box[3]))
    box_width  = int(np.linalg.norm(box[0] - box[1]))

    if box_height > 2 * box_width and box_height > 30:
        img_text  = Image.new("RGB", (box_height, box_width), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        font = create_font(txt, (box_height, box_width), font_path)
        draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
        img_text = img_text.transpose(Image.ROTATE_270)
    else:
        img_text  = Image.new("RGB", (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        font = create_font(txt, (box_width, box_height), font_path)
        draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
    
    pts1 = np.float32([[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]])
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_text = np.array(img_text, dtype=np.uint8)
    img_right_text = cv2.warpPerspective(
        img_text,
        M,
        img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return img_right_text        

def draw_ocr_box_txt(image, boxes, txts, scores, font_path=None, drop_score=0.5):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    h, w  = image.height, image.width
    img_left  = image.copy()
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    random.seed(0)

    draw_left = ImageDraw.Draw(img_left)
    if txts is None or len(txts) != len(boxes):
        txts = [None] * len(boxes)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw_left.polygon(box, fill=color)
        img_right_text = draw_box_txt((w, h), box, txt, font_path)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_right_text, [pts], True, color, 1)
        img_right = cv2.bitwise_and(img_right, img_right_text)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    return np.array(img_show)

if __name__ == "__main__":

    image = cv2.imread("deploy/lite/imgs/lite_demo.png")
    src_h, src_w, _ = image.shape

    det_model_file_path = "models/det/det.sim.onnx"
    cls_model_file_path = "models/cls/cls.sim.onnx"
    rec_model_file_path = "models/rec/rec.sim.onnx"
    character_dict_path = "ppocr/utils/ppocr_keys_v1.txt"
    font_path = "doc/fonts/simfang.ttf"

    # 1. text detection
    text_detector = TextDetector(det_model_file_path)
    img_pre   = text_detector.preprocess(image)
    det_pred  = text_detector.forward(img_pre)
    det_boxes = text_detector.postprocess(det_pred, src_h, src_w)

    # 2. text classification
    if det_boxes is None:
        print("warning, no det_boxes found")
        exit()
    else:
        print(f"det_boxes num: {len(det_boxes)}")
    text_classifier = TextClassifier(cls_model_file_path)
    img_list, imgs_pre_batch, indices = text_classifier.preprocess(image, det_boxes)
    img_list, _ = text_classifier.postprocess(img_list, imgs_pre_batch, indices)

    # 3. text recognition
    text_recognizer = TextRecognizer(rec_model_file_path, character_dict_path)
    imgs_pre_batch, indices = text_recognizer.preprocess(img_list)
    rec_txts = text_recognizer.postprocess(imgs_pre_batch, indices)

    # 4. visualization
    txts   = [rec_txts[i][0] for i in range(len(rec_txts))]
    scores = [rec_txts[i][1] for i in range(len(rec_txts))]
    
    draw_img = draw_ocr_box_txt(image, det_boxes, txts, scores, font_path)
    cv2.imwrite("result.jpg", draw_img[:, :, ::-1])