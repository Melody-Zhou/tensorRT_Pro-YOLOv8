from ultralytics import YOLO

model = YOLO("yolov8s.pt")

success = model.export(format="onnx", dynamic=True, simplify=True)