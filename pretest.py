import os
import cv2
import matplotlib.pyplot as plt

radar_path = "/home/wei/Radar_Object_Detector/data/training_data_2/city_1_3/images"
txt_path = "/home/wei/Radar_Object_Detector/data/training_data_2/city_1_3/labels"

image_files = sorted([f for f in os.listdir(radar_path) if f.endswith('.png')])

for image_file in image_files:

    image_path = os.path.join(radar_path, image_file)
    label_file = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(txt_path, label_file)

    if os.path.exists(label_path):

        image = cv2.imread(image_path)
        
        with open(label_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            data = line.split()

            object_class = int(data[0])
            x_center = float(data[1])*1152
            y_center = float(data[2])*1152
            width = float(data[3])*1152
            height = float(data[4])*1152

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            color = (0, 255, 0)  
            if object_class == 2:
                color = (0, 0, 255)  

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        #time.sleep(0.1)

    cv2.destroyAllWindows()
