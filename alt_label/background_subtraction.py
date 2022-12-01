import cv2
import math
import numpy as np
def main():
# Create tracker object
    tracker = EuclideanDistTracker()

    video_path = "mov_4.MOV"
    cap = cv2.VideoCapture(video_path)
    
# Object detection from Stable camera
    object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=100, detectShadows=False)


# Speed Parameters
    #scale = distance(, ) / 2



    while cap.isOpened():
        input("Press Enter to continue...")
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 360), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        height, width, _  = frame.shape


        #print(height)
        #print(width)
    # Extract Region of interest
        roi = frame[125: height,0: width]

    # 1. Object Detection
        mask = object_detector.apply(roi)
        #_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        mask = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        #mask = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
        # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(cnt)


                detections.append([x, y, w, h])

    # 2. Object Tracking
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            if (w > 80 or h > 60) and (w < 400 and h < 300):
                cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                print("Image - Tracker ID: " + str(id) + " speed: " + str(tracker.curr_id_speeds[id]))
                #if id == 32:
                #    print(boxes_ids)
                #    input("pause here")
                #print(tracker.curr_id_speeds[id])
                #cv2.putText(roi, str(tracker.curr_id_speeds[id]), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

        #cv2.imshow("roi", roi)
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(30)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



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
                #print(type(self.prev_center_points))
                #print(obj_bb_id)
                #print(type(obj_bb_id))
                #print(self.prev_center_points)
                if object_id in self.prev_center_points: #if(len(self.prev_center_points)>0):
                #print(center)
                    curr_center = []
                    prev_center = []
                    curr_center.append(center[0])
                    curr_center.append(center[1])
                #print(self.prev_center_points[object_id])
                    prev_center.append(self.prev_center_points[object_id][0])
                    prev_center.append(self.prev_center_points[object_id][1]) 
            ### CALCULATE SPEED ###
            #scale = height 
                    scale = h/2 #speed will be calculated with assumption that average height of bounding box is ~2meters
                    dist = distance(prev_center, curr_center)/scale
                    speed = int(dist/.033) #assuming 30fps
                    if h > 60 or w > 80:
                    #    print("current coord")
                    #    print(curr_center)
                    #    print("prev coord")
                    #    print(prev_center)
                        if(speed > 0):
                            print("object id: " + str(object_id) + " speed: " + str(speed))
                            self.curr_id_speeds[object_id] = speed
		    #new_speeds[object_id] = speed



        # Update dictionary with IDs not used removed
        #self.curr_id_speeds = new_speeds.copy()
        self.center_points = new_center_points.copy()
        self.prev_center_points = new_center_points.copy()
        
        return objects_bbs_ids

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)



if __name__ == "__main__":
    main()
