from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="/home/wei/Radar_Object_Detector/training.yaml", imgsz=480, epochs=4000, batch=-1, device=0, workers = 12)  # train the model
model.export(format="onnx")