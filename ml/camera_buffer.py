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

    def __init__(self, MAX_BUFFER_LEN=200):
        # Recent frames are at very end of buffer
        self.MAX_BUFFER_LEN = MAX_BUFFER_LEN
        self.image_buffer = []
        self.buffer_lock = threading.Lock()
        self.thread = threading.Thread(target=self.read_camera_thread, daemon=False)
        self.thread.start()
        self.batch_idx = 0

    def read_camera_thread(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        try:
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
                processed = image_preprocess(image, IMAGE_SIZE, 0, 1)[0]
                self.image_buffer.append(tf.cast(processed, tf.uint8))
                self.buffer_lock.release()
        except Exception as e:
            cap.release()
            raise e

    def get_recent_batch(self, batch_size=8):
        self.buffer_lock.acquire()
        batch = tf.stack(self.image_buffer[-batch_size:])
        self.batch_idx = len(self.image_buffer)
        self.buffer_lock.release()
        return batch

    def get_matching_buffer(self, batch_size=8):
        self.buffer_lock.acquire()
        if self.batch_idx <= 0:
            print("ERROR: Trying to retrieve matching buffer for a batch that has gone completely from the buffer")
        else:
            stream = np.stack(self.image_buffer[self.batch_idx:self.batch_idx + batch_size])
        self.buffer_lock.release()
        return stream


if __name__ == '__main__':
    c = CameraBuffer(400)
    time.sleep(10)
    print(len(c.image_buffer))
    batch = c.get_recent_batch()
    print(batch.shape)
    for b in batch:
        print(b.shape, b.dtype)
    time.sleep(100)
