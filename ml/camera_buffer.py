import threading
import cv2
import time
import sys
import os
import numpy as np
import tensorflow.compat.v1 as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "../automl/efficientdet"))
from inference import image_preprocess

IMAGE_SIZE = (320, 320)

class CameraBuffer:

    def __init__(self, MAX_BUFFER_LEN=100):
        # Recent frames are at very end of buffer
        self.MAX_BUFFER_LEN = MAX_BUFFER_LEN
        self.image_buffer = []
        self.buffer_lock = threading.Lock()
        self.thread = threading.Thread(target=self.read_camera_thread, daemon=False)
        self.batch_idx = 0

    def read_camera_thread(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera failed to read")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.buffer_lock.acquire()
            if len(self.image_buffer) >= self.MAX_BUFFER_LEN:
                self.image_buffer.pop(0)
                self.batch_idx -= 1
            self.image_buffer.append(image_preprocess(image, IMAGE_SIZE, 0, 1))
            self.buffer_lock.release()

    def get_recent_batch(self, batch_size=8):
        self.buffer_lock.acquire()
        batch = tf.stack(self.image_buffer[-batch_size:])
        self.batch_idx = len(self._image_buffer)
        self.buffer_lock.release()
        return batch

    def get_matching_buffer(self):
        self.buffer_lock.acquire()
        if self.batch_idx <= 0:
            print("ERROR: Trying to retrieve matching buffer for a batch that has gone completely from the buffer")
        else:
            history = np.stack(self.image_buffer[:self.batch_idx])
        self.buffer_lock.release()
        return history

