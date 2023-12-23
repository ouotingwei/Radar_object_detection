from ultralytics import YOLO

# Load a model
model = YOLO("yolov8.yaml")  # build a new model from scratch

# Use the model
model.train(data="/home/wei/Radar_Object_Detector/training.yaml", imgsz=580, epochs=1000, batch=70, device=0, workers=6)  # train the model
model.export(format="onnx")