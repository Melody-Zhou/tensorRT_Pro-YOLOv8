import cv2
from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolov8s-cls.pt")

    img = cv2.imread("ultralytics/assets/bus.jpg")

    result = model(img)[0]
    names  = result.names

    top1_label = result.probs.top1
    top5_label = result.probs.top5
    top1_conf  = result.probs.top1conf
    top5_conf  = result.probs.top5conf
    top1_name  = names[top1_label]

    print(f"The model predicted category is {top1_name}, label = {top1_label}, confidence = {top1_conf:.4f}")