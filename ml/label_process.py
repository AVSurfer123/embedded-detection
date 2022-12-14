"""
Run object detection on images, Press ESC to exit the program
For Raspberry PI, please use `import tflite_runtime.interpreter as tflite` instead
"""
import re
import cv2
import numpy as np
import math
import os
import sys

import tensorflow as tf
import tensorflow.lite as tflite
# import tflite_runtime.interpreter as tflite
from statistics import mode
from PIL import Image
from classify_bb import *
from camera_buffer import CameraBuffer

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 320
NUM_BATCH = 8
sys.path.append(os.path.join(os.path.dirname(__file__), "../automl/efficientdet"))
from run_tflite import TFLiteRunner
sys.path.append(os.path.join(os.path.dirname(__file__), "../networking"))
from data_utils import write_image, load_model

def save_visualized_image(image, prediction, old_c_array):
    """Saves the visualized image with prediction.

    Args:
        image: numpy.ndarray of shape [H, W, C].
        prediction: numpy.ndarray of shape [num_predictions, 7].
        output_path: str, output image path.
    """
  #[image_id, ymin, xmin, ymax, xmax, score, class]

    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.6
    color = (255, 0, 0)  # Blue color
    thickness = 1
    batch_array = []
    batch_num = 0
    for batch in prediction:
        new_c_array = []
        for result in batch:
            if result[5] > .3:
                image_id = int(result[0])
                y1 = result[1]
                x1 = result[2]
                y2 = result[3]
                x2 = result[4]
                score = result[5]
                label_id = result[6]
                cx = (x2-x1)/2 + x1
                cy = (y2-y1)/2 + y1
                w = x2 - x1
                h = y2 - y1
                if (old_c_array[batch_num].count((cx, cy))) == 0 and (h > 60 or w > 80):
                    new_c_array.append((cx, cy))
                    if len(image[image_id]) != 0:
                        cv2.putText(image[image_id].numpy(), str(label_id), (int(x1), int(y1)), font, size, color, thickness)
                        cv2.rectangle(image[image_id].numpy(), (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        # if len(image[batch_num]) > 0:
        #     cv2.imshow("Object Detect", image[batch_num].numpy())
        # key = cv2.waitKey(1)
        # if key == 27:  # esc
        #     break
        batch_array.append(new_c_array)
        batch_num = batch_num + 1
    return(batch_array)
            

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        self.prev_center_points = {}
        self.curr_id_speeds = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1


        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        #new_speeds = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, w, h, object_id = obj_bb_id
            #new_speeds[object_id] = 0
            if object_id not in new_center_points:
                self.curr_id_speeds[object_id] = 0
                center = self.center_points[object_id]
                new_center_points[object_id] = center
                if object_id in self.prev_center_points: #if(len(self.prev_center_points)>0):
                    curr_center = []
                    prev_center = []
                    curr_center.append(center[0])
                    curr_center.append(center[1])
                    prev_center.append(self.prev_center_points[object_id][0])
                    prev_center.append(self.prev_center_points[object_id][1])
                    ### CALCULATE SPEED ###
                    scale = h/2 #speed will be calculated with assumption that average height of bounding box is ~2meters
                    dist = distance(prev_center, curr_center)/scale
                    speed = dist*30 #assuming 30fps video feed
                    if h > 60 or w > 80:
                        if(speed > 0):
                            self.curr_id_speeds[object_id] = speed
        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        self.prev_center_points = new_center_points.copy()

        return objects_bbs_ids

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def create_JSON(bbox, alt_label):
    bbox_p_label = [int(bbox[1]),int( bbox[2]), int(bbox[3]), int(bbox[4]), alt_label]
    return bbox_p_label
    


if __name__ == "__main__":

    video_path = "data/IMG_1702.MOV"     # good_example vid = variety_lens_flare.MOV"
    cap = cv2.VideoCapture(video_path)
    if(cap.isOpened() == False):
        print("VID DID NOT OPEN")
    c_array = [[] for _ in range(NUM_BATCH)]
    runner = TFLiteRunner("../automl/efficientdet/efficientdet-batch8.tflite")
    #camera_cap = CameraBuffer()   
    tracker = EuclideanDistTracker()
    object_detector = cv2.createBackgroundSubtractorMOG2(history = 200, varThreshold = 100, detectShadows=False)


    print("Beginning processing")
    import time
    time.sleep(2)
    # Process Stream
    while True:
        #image = camera_cap.get_recent_batch(NUM_BATCH)
        #USED FOR DEBUG VIDEO STREAM#
        
        img_arr = []
        for i in range(0, NUM_BATCH):
            ret, frame = cap.read()             
            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_LINEAR)
        #ret, frame = cap.read()
        #frame = cv2.resize(frame, (640, 360), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
            single_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            single_image = single_image.resize((CAMERA_WIDTH, CAMERA_HEIGHT))
            img_arr.append(np.array(single_image))
        image = tf.convert_to_tensor(img_arr, dtype=tf.uint8)
        #manually batch the test video

        prediction = runner.run(image)
        new_c_array = save_visualized_image(image, prediction, c_array)
        
        c_array = new_c_array
        # input("pause")
        for batch_num in range(0, NUM_BATCH):
            if len(new_c_array[batch_num]) == 1:
            #If the ML model only detects one, object, begin tracking that object with alternative labeling
            #display_result(top_result, frame, labels)
             #roi = frame[0: height, 0: width]
                ml_labels = []
                for result in prediction[batch_num]:
                    if result[5] > .3:
                        image_id = result[0]
                        y1 = result[1]
                        x1 = result[2]
                        y2 = result[3]
                        x2 = result[4]
                        score = result[5]
                        ml_label = result[6]
                        cx = (x2-x1)/2 + x1
                        cy = (y2-y1)/2 + y1
                        w = x2 - x1
                        h = y2 - y1
                        if cx == new_c_array[batch_num][0][0] and cy == new_c_array[batch_num][0][1]:
                            ml_labels.append(ml_label)
                            print("ML Label: " + str(ml_label))
                            # input("pause until enter pressed:")
                    alt_labels = []
                #GRAB NEXT BATCH OF IMAGES*****#
                img_arr = []
                '''for i in range(0, NUM_BATCH):
                    ret, frame = cap.read()             
                    frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        #ret, frame = cap.read()
        #frame = cv2.resize(frame, (640, 360), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
                    single_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    single_image = image.resize((CAMERA_WIDTH, CAMERA_HEIGHT))
                    img_arr.append(single_image)
                next_batch_image = tf.stack([img_arr[0], img_arr[1], img_arr[2], img_arr[3], img_arr[4], img_arr[5], img_arr[6], img_arr[6]])
'''
                #next_batch_image = camera_cap.getMatchingBuffer(NUM_BATCH) 
                for i in range(0,NUM_BATCH): 
                    ret, frame = cap.read()
                    #frame = next_batch_image[i]
                    frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_LINEAR)
                    #frame = images
                    #object detection
                    mask = object_detector.apply(frame)
                    mask = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    detections = []
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > 100:
                            x, y, w, h = cv2.boundingRect(cnt)
                    
                            detections.append([x, y, w, h])
                 
                    boxes_ids = tracker.update(detections)
                    for box_id in boxes_ids:
                        x, y, w, h, id = box_id
                        if(w > 80 or h > 60) and (w < 400 and h < 300):
                            alpha = h/w
                            alt_label = classify_bb(tracker.curr_id_speeds[id], alpha)
                            alt_labels.append(classify_bb(tracker.curr_id_speeds[id], alpha))
                            cv2.putText(frame, str(alt_label), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    # cv2.imshow("Object Detection", frame)
                    # cv2.imshow("Mask", mask)
                    # key = cv2.waitKey(1)
                    # if key == 27:  # esc
                    #     break


                    # input("Pause until enter")
                print(alt_labels)
                alt_label = mode(alt_labels)
            
                if alt_label not in ml_labels:
                    #return image = image[batch_num]
                    #prediction[batch_num][0]
                    return_img = image[batch_num]
                    return_list = create_JSON(prediction[batch_num][0], alt_label)
                    print(return_list)
                    write_image(return_img, return_list)
                    print("ML Label: " + str(ml_label) + " ALT Label: " + str(alt_label))
                    print(ml_labels)
                    # input("enter")
                else:
                    print("ML Label and Alt Label Match")

