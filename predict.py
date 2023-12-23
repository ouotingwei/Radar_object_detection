import cv2
from ultralytics import YOLO
import os
import numpy as np
import radiate

testing_data_dir = '/home/wei/Radar_Object_Detector/data/testing_data/city_7_0'
congig_file_dir = '/home/wei/Radar_Object_Detector/config.yaml'

dt = 0.25
# load sequence
seq = radiate.Sequence(testing_data_dir, congig_file_dir)

def main():

    model = YOLO('/home/wei/Radar_Object_Detector/training_results/12_20_all/detect/train6/weights/best.pt')

    image_folder = '/home/wei/Radar_Object_Detector/data/testing_data/city_7_0/images'

    camera_folder = '/home/wei/Radar_Object_Detector/data/testing_data/city_7_0/zed_right'

    cv2.namedWindow("YOLOv8", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8", 1080, 1080)

    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]

    images.sort()

    for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
        output = seq.get_from_timestamp(t)
        if output != {}:
            radar = output['sensors']['radar_cartesian']
            camera = output['sensors']['camera_right_rect']
            
            print(output)
            '''
            results = model(frame)
            annotated_frame = results[0].plot()

            for r in results:
                obj_amount = len(r.boxes.xywh)
                
                if obj_amount != 0:
                    for id in range(obj_amount):
                        x = r.boxes.xywh[id][0]
                        y = r.boxes.xywh[id][1]
                


                
                objects.append({'bbox': {'position': ??}, 'class_name': 'moving object'})

            radar = seq.vis(radar, objects, color=(255,0,0))

            bboxes_cam = seq.project_bboxes_to_camera(objects,
                                                    seq.calib.right_cam_mat,
                                                    seq.calib.RadarToRight)
            # camera = seq.vis_3d_bbox_cam(camera, bboxes_cam)
            camera = seq.vis_bbox_cam(camera, bboxes_cam)

            cv2.imshow('radar', radar)
            cv2.imshow('camera_right_rect', camera)
            # You can also add other sensors to visualize
            cv2.waitKey(1)
            '''
            
    '''
    for image_file in images:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)

        if frame is not None:
            results = model(frame)
            annotated_frame = results[0].plot()

            for r in results:
                obj_amount = len(r.boxes.xywh)
                
                if obj_amount != 0:
                    for id in range(obj_amount):
                        x = r.boxes.xywh[id][0]
                        y = r.boxes.xywh[id][1]
                    

            cv2.imshow("YOLOv8", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(1)
        else:
            print(f"Unable to read image: {image_file}")

    cv2.destroyAllWindows()
    '''

if __name__ == '__main__':
    main()