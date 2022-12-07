import sys
import time
import logging
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.lite as tflite

from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler

class MyHandler(LoggingEventHandler):
    def on_created(self, event):
        super().on_created(event)
        what = 'directory' if event.is_directory else 'file'
        self.logger.info("Created %s: %s", what, event.src_path)
        if True: # Replace w/ condition to retrain; e.g. every 5 min?
            retrain_model()

# Use Watchdog to see when pred/ updates with .npy files
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    path = sys.argv[1] if len(sys.argv) > 1 else '.'

    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()

def retrain_model():
    pass
    # data = np.load('pred/camera/*')

    # detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    # input_sample = input[0, :, :, :][np.newaxis, :, :, :]
    # detector_output = detector(input_sample)