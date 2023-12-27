import os
import json
import math
import numpy as np

def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
        print(f"label folder {folder_path} has been created.")
    except FileExistsError:
        print(f"label folder {folder_path} has already been created.")


def gen_boundingbox(bbox, angle):
        theta = np.deg2rad(-angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        points = np.array([[bbox[0], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                           [bbox[0], bbox[1] + bbox[3]]]).T

        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        T = np.array([[cx], [cy]])

        points = points - T
        points = np.matmul(R, points) + T
        points = points.astype(int)

        min_x = np.min(points[0, :])
        min_y = np.min(points[1, :])
        max_x = np.max(points[0, :])
        max_y = np.max(points[1, :])

        return min_x, min_y, max_x, max_y


def create_label(radar_folder_path, label_path, txt_path):
    """
    convert radar label to yolo label
    """

    cls = ["car", "bus", "van", "truck", "pedestrian", "group_of_pedestrians"]

    image_extensions = ['.png']

    if not os.path.exists(radar_folder_path):
        print(f"folder {radar_folder_path} doesn't exist")
        return

    # open the .json file
    with open(label_path, 'r') as f_annotation:
        annotation = json.load(f_annotation)

    # create yolo label.txt
    obj_cnt = 0

    # Sort the list of filenames
    sorted_filenames = sorted(os.listdir(radar_folder_path))

    for file_name in sorted_filenames:
        base_name = os.path.splitext(file_name)[0]

        # create .txt
        txt_file_path = os.path.join(txt_path, f"{base_name}.txt")

        with open(txt_file_path, 'w') as txt_file:

            for obj_dict in annotation:

                current_id = obj_dict.get("id")

                if current_id is not None and obj_dict["class_name"] != cls[4] and obj_dict["class_name"] != cls[5]:
                    #print(current_id)

                    # Ensure obj_cnt is within the range of bboxes
                    if obj_cnt < len(obj_dict.get("bboxes", [])):

                        bbx = obj_dict["bboxes"][obj_cnt]
                        
                        if "position" in bbx:

                            rotation = bbx["rotation"]

                            bx = [bbx["position"][0], bbx["position"][1], bbx["position"][2], bbx["position"][3]]

                            min_x, min_y, max_x, max_y = gen_boundingbox(bx, rotation)

                            center_x = (max_x + min_x) / 2
                            center_y = (max_y + min_y) / 2

                            modify_w = (max_x - min_x)
                            modify_h = (max_y - min_y)

                            # Set class ID to 0 for all objects
                            obj_dict["class_name"] = 0

                            txt_file.write(f"{0} {center_x/1152} {center_y/1152} {modify_w/1152} {modify_h/1152}\n")
                    else:
                        print(f"No bbox found for Object ID {current_id}")

        obj_cnt += 1

def main():

    # training data
    label_path = "/home/wei/Radar_Object_Detector/data/training_data_2/rain_4_1/annotations/annotations.json"
    radar_path = "/home/wei/Radar_Object_Detector/data/training_data_2/rain_4_1/images"
    txt_path = "/home/wei/Radar_Object_Detector/data/training_data_2/rain_4_1/labels"

    create_folder(txt_path)

    create_label(radar_path, label_path, txt_path)


if __name__ == '__main__':
    main()
