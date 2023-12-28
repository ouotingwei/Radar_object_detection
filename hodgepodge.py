import os
import shutil

target_img_dir = "/home/wei/Radar_Object_Detector/data/whole_train/images"
target_label_dir = "/home/wei/Radar_Object_Detector/data/whole_train/labels"

target_img_files = sorted([f for f in os.listdir(target_img_dir) if f.endswith(".png")])
target_label_files = sorted([f for f in os.listdir(target_label_dir) if f.endswith(".txt")])

# 更改這裡的路徑
input_img_dir = "/home/wei/Radar_Object_Detector/data/training_data_2/rain_4_1/images"
input_label_dir = "/home/wei/Radar_Object_Detector/data/training_data_2/rain_4_1/labels"

img_files = sorted([f for f in os.listdir(input_img_dir) if f.endswith(".png")])
label_files = sorted([f for f in os.listdir(input_label_dir) if f.endswith(".txt")])

target_img_len = len(target_img_files)
target_label_len = len(target_label_files)

# 使用新的計數器
file_cnt = target_img_len

for i, (label_file, image_file) in enumerate(zip(label_files, img_files)):
    label_src = os.path.join(input_label_dir, label_file)
    image_src = os.path.join(input_img_dir, image_file)

    # 新的檔案名稱
    new_filename = f"{i + 1 + target_img_len:07d}"

    label_dst = os.path.join(target_label_dir, f"{new_filename}.txt")
    image_dst = os.path.join(target_img_dir, f"{new_filename}.png")

    # 輸出語句
    print(f"Copying {label_src} to {label_dst}")
    print(f"Copying {image_src} to {image_dst}")

    # 複製檔案
    shutil.copy(label_src, label_dst)
    shutil.copy(image_src, image_dst)

    # 更新計數器
    file_cnt += 1

print(f"檔案合併完成，總共 {file_cnt} 個檔案。")
