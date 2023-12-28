#!/usr/bin/env python3

import cv2
from ultralytics import YOLO
import os
import time
import json
import numpy as np

def float32_handler(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def main():

    model = YOLO('/home/wei/Radar_Object_Detector/training_results/12_27_v8m/detect/train2/weights/best.pt')

    image_folder = '/home/wei/Radar_Object_Detector/data/testing_data/city_7_0/images'

    output_json_path = '/home/wei/Radar_Object_Detector/predict_results/1228.json'

    cv2.namedWindow("YOLOv8 Predictor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 Predictor", 1000, 1000)

    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]

    images.sort()

    data_list = []

    for image_file in images:
        print(image_file)
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)

        if frame is not None:
            results = model(frame)

            sample_token, _ = os.path.splitext(image_file)

            for r in results:
                points = []
                for box in r.boxes.xyxy:
                    x_min, y_min, x_max, y_max = box.detach().cpu().numpy()
                    print(sample_token, x_min, y_min, x_max, y_max)
                    points.append([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]])

                    annotated_frame = r.plot()
                    cv2.imshow("YOLOv8 Predictor", annotated_frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                    time.sleep(0.2)

                    data_list.append({
                        "sample_token": sample_token,
                        "points": points,
                        "name": "car"
                    })
        else:
            print(f"Unable to read image: {image_file}")

    cv2.destroyAllWindows()

    with open(output_json_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=4, default=float32_handler)

if __name__ == '__main__':
    main()
