import cv2
import numpy as np
debug = False
    
def capture_video():
    # Setup Video/Webcam Stream
    vid = cv2.VideoCapture('sample.MOV')
    if (vid.isOpened()== False): 
        print("Error opening video stream or file")
    
    # Read until video is completed
    frames = []
    while(vid.isOpened()):
        ret, frame = vid.read()
        if ret == True:
            cv2.imshow('Frame', frame)
            frames.append(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break

    vid.release() 
    cv2.destroyAllWindows()

    # return np.array(frames, dtype=np.int8) # ISSUE: RETURN IS TAKING TOO LONG?
    return np.array(frames)

def capture_webcam():
    vid = cv2.VideoCapture(0)
    if (vid.isOpened()== False): 
        print("Error opening video stream or file")

    while(True):
        ret, frame = vid.read()
        if ret and debug:
            print("OK")
    
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()