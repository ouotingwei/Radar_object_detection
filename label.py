
"""
@author: OU,TING-WEI @ Graduate Degree Program in Robotics ( NYCU FALL-2023 )
@date: 2023-12-16
@contact: kklb1716@gmail.com
@description: Radar object detection w/ YOLO V8
"""

import os
import json

def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
        print(f"label folder {folder_path} has been created.")
    except FileExistsError:
        print(f"label folder {folder_path} has already been created.")

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

                if current_id is not None:
                    #print(f"Object ID: {current_id}")

                    # Ensure obj_cnt is within the range of bboxes
                    if obj_cnt < len(obj_dict.get("bboxes", [])):

                        bbx = obj_dict["bboxes"][obj_cnt]
                        
                        if "position" in bbx:

                            x = bbx["position"][0]
                            y = bbx["position"][1]
                            width = bbx["position"][2]
                            height = bbx["position"][3]

                            x_center = ( x + width / 2 ) / 1152
                            y_center = ( y + height / 2 ) / 1152

                            width = width / 1152
                            height = height / 1152

                            for i in range(len(cls)):
                                if obj_dict["class_name"] == cls[i]:
                                    obj_dict["class_name"] = i

                            txt_file.write(f"{0} {x_center} {y_center} {width} {height}\n")
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