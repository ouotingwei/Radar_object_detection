from ultralytics import YOLO

# Load a model
model = YOLO("/home/wei/Radar_Object_Detector/training_results/0101_2259/detect/0101_2259/weights/last.pt")  # build a new model from scratch

# Use the model
model.train(data="/home/wei/Radar_Object_Detector/training.yaml", imgsz=480, epochs=1000, batch=-1, device=0, workers=8)  # train the model
model.export(format="onnx")