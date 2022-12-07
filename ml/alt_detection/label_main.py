"""
Run object detection on images, Press ESC to exit the program
For Raspberry PI, please use `import tflite_runtime.interpreter as tflite` instead
"""
import re
import cv2
import numpy as np
import math

import tensorflow.lite as tflite
# import tflite_runtime.interpreter as tflite
from statistics import mode
from PIL import Image
from classify_bb import *

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360


def load_labels(label_path):
    r"""Returns a list of labels"""
    with open(label_path) as f:
        labels = {}
        for line in f.readlines():
            m = re.match(r"(\d+)\s+(\w+)", line.strip())
            labels[int(m.group(1))] = m.group(2)
        return labels


def load_model(model_path):
    r"""Load TFLite model, returns a Interpreter instance."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def process_image(interpreter, image, input_index):
    r"""Process an image, Return a list of detected class ids and positions"""
    input_data = np.expand_dims(image, axis=0)  # expand to 4-dim

    # Process
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Get outputs
    output_details = interpreter.get_output_details()
    # print(output_details)
    # output_details[0] - position
    # output_details[1] - class id
    # output_details[2] - score
    # output_details[3] - count

    positions = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))

    result = []

    for idx, score in enumerate(scores):
        if score > 0.5:
            result.append({'pos': positions[idx], '_id': classes[idx]})

    return result


def display_result(result, frame, labels, old_c_array):
    r"""Display Detected Objects"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.6
    color = (255, 0, 0)  # Blue color
    thickness = 1
    new_c_array = []
    # position = [ymin, xmin, ymax, xmax]
    # x * CAMERA_WIDTH
    # y * CAMERA_HEIGHT
    for obj in result:
        pos = obj['pos']
        _id = obj['_id']

        x1 = int(pos[1] * CAMERA_WIDTH)
        x2 = int(pos[3] * CAMERA_WIDTH)
        y1 = int(pos[0] * CAMERA_HEIGHT)
        y2 = int(pos[2] * CAMERA_HEIGHT)
        cx = (x2-x1)/2 + x1
        cy = (y2-y1)/2 + y1
        w = x2 - x1
        h = y2 - y1
        if old_c_array.count((cx, cy)) == 0 and (h > 60 or w > 80):
            new_c_array.append((cx, cy))
            cv2.putText(frame, labels[_id], (x1, y1), font, size, color, thickness)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow('Object Detection', frame)

    return new_c_array


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
                    #print(self.center_points)
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


if __name__ == "__main__":

    model_path = 'data/detect.tflite'
    label_path = 'data/coco_labels.txt'
    video_path = "data/1_skateboard.MOV"     # good_example vid = variety_lens_flare.MOV"
    cap = cv2.VideoCapture(video_path)
    
    #cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    #cap.set(cv2.CAP_PROP_FPS, 30)

    c_array = []

    interpreter = load_model(model_path)
    labels = load_labels(label_path)

    input_details = interpreter.get_input_details()
    
    tracker = EuclideanDistTracker()
    object_detector = cv2.createBackgroundSubtractorMOG2(history = 200, varThreshold = 100, detectShadows=False)


    # Get Width and Height
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]

    # Get input index
    input_index = input_details[0]['index']
    print("Beginning processing")
    # Process Stream
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 360), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        #ret, frame = cap.read()
        #frame = cv2.resize(frame, (640, 360), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((width, height))

        top_result = process_image(interpreter, image, input_index)
        new_c_array = display_result(top_result, frame, labels, c_array)
        c_array = new_c_array
        input("pause")
        if len(new_c_array) == 1:
	    #If the ML model only detects one, object, begin tracking that object with alternative labeling
	    #display_result(top_result, frame, labels)
            #roi = frame[0: height, 0: width]
            ml_labels = []
            for obj in top_result:
                pos = obj['pos']
                x1 = int(pos[1] * CAMERA_WIDTH)
                x2 = int(pos[3] * CAMERA_WIDTH)
                y1 = int(pos[0] * CAMERA_HEIGHT)
                y2 = int(pos[2] * CAMERA_HEIGHT)
                cx = (x2-x1)/2 + x1
                cy = (y2-y1)/2 + y1
                if cx == new_c_array[0][0] and cy == new_c_array[0][1]:
                    ml_label = obj['_id']
                ml_labels.append(obj['_id'])
            print("ML Label: " + str(ml_label))
            input("pause until enter pressed:")
            alt_labels = []

            for i in range(0,10):
                ret, frame = cap.read()
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
                cv2.imshow("Object Detection", frame)
                cv2.imshow("Mask", mask)
                input("Pause until enter")
            print(alt_labels)
            alt_label = mode(alt_labels)
            
            if alt_label not in ml_labels:
                print("ML Label: " + str(ml_label) + " ALT Label: " + str(alt_label))
                print(ml_labels)
                input("enter")
            else:
                print("ML Label and Alt Label Match")



        key = cv2.waitKey(1)
        if key == 27:  # esc
            break

    cap.release()
    cv2.destroyAllWindows()
