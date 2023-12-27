from ultralytics import YOLO

# Load a model
model = YOLO("yolov8l.yaml")  # build a new model from scratch

# Use the model
model.train(data="/home/wei/Radar_Object_Detector/training.yaml", imgsz=576, epochs=150, batch=-1, device=0, workers = 12)  # train the model
model.export(format="onnx")