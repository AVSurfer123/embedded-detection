import numpy as np
from skimage import io
import tensorflow as tf
import os
import time
import json
from typing import List, Tuple, Any, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(CURRENT_DIR, "images")
LABEL_DIR = os.path.join(CURRENT_DIR, "labels")
WEIGHT_DIR = os.path.join(CURRENT_DIR, "weights")

SERVER_IMAGE_DIR = os.path.join(CURRENT_DIR, "server_images")
SERVER_LABEL_DIR = os.path.join(CURRENT_DIR, "server_labels")
SERVER_WEIGHT_DIR = os.path.join(CURRENT_DIR, "server_weights")

IMAGE_READ_DELAY = 1.0
WEIGHT_READ_DELAY = 1.0

def read_new_images(last_time: float) -> Tuple[List[Tuple[np.ndarray, Any]], float]:
    """
    Server method: Reads new images and labels from disk which have been sent by edge clients.

    Args:
        last_time: Only gets images/labels in disk after this time. Represented as seconds after Unix epoch, like time.time()

    Returns:
        Tuple of 
            List of new data tuples. Each tuple has length 2 consisting of (img, label) 
                where img is an np array in RGB color space (h, w, 3)
                and label is a JSON object of the necessary labels for training
            New last_time to use, which is the latest image read from disk
    """
    os.makedirs(SERVER_IMAGE_DIR, exist_ok=True)
    os.makedirs(SERVER_LABEL_DIR, exist_ok=True)
    data = []
    max_time = last_time
    cur_time = time.time()
    for file_name in os.listdir(SERVER_IMAGE_DIR):
        file_path = os.path.join(SERVER_IMAGE_DIR, file_name)
        # mtime = os.stat(file_path).st_mtime
        mtime = int(file_name.split('.')[0]) / 1000 
        if mtime > last_time and (cur_time - os.stat(file_path).st_mtime) > IMAGE_READ_DELAY:
            # img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = io.imread(file_path)
            label_path = os.path.join(SERVER_LABEL_DIR, f"{file_name.split('.')[0]}.txt")
            with open(label_path, 'r') as f:
                label = json.load(f)
            data.append((img, label))
            max_time = max(max_time, mtime)
    return data, max_time

def write_image(image: np.ndarray, label: Any):
    """
    Client method: Saves an image and label data to disk which will be sent to the central server.

    Args:
        image: np array in RGB color space (h, w, 3)
        label: JSON object with necessary labels for training
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)
    time_ms = time.time_ns() // 1_000_000 
    with open(os.path.join(LABEL_DIR, f"{time_ms}.txt"), 'w') as f:
        json.dump(label, f, sort_keys=True, indent=2)
    # cv2.imwrite(os.path.join(IMAGE_DIR, f"{time_ms}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    io.imsave(os.path.join(IMAGE_DIR, f"{time_ms}.png"), image)

def load_model(last_time: float) -> Tuple[Optional[tf.lite.Interpreter], float]:
    """
    Client method: Loads weights from disk into the model, if newer weights have arrived from the central server.

    Args:
        last_time: Only loads the weights if they arrived after this time. Represented as seconds after Unix epoch, like time.time()

    Returns:
        Tuple of
            TF Lite Interpreter for new model that is loaded or None if no new model was found,
            New last_time to use which is modification time of model on the disk
    """
    file_name = os.path.join(WEIGHT_DIR, "model.tflite")
    if not os.path.exists(file_name):
        return None, last_time
    mtime = os.stat(file_name).st_mtime
    if mtime > last_time and (time.time() - mtime) > WEIGHT_READ_DELAY:
        interpreter = tf.lite.Interpreter(file_name)
        return interpreter, mtime
    return None, last_time

def save_model(tflite_model):
    """
    Server method: Saves the given TFLite model to the disk to send to the edge clients.

    Args:
        model: TFLite model to save
    """
    os.makedirs(SERVER_WEIGHT_DIR, exist_ok=True)
    with open(os.path.join(SERVER_WEIGHT_DIR, "model.tflite"), 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    print('hello i started')
    # image = io.imread("/home/ubuntu/embedded-detection/image.png")
    # write_image(image, {'testing': 'the json', 'of': 123, 'dreams': np.random.rand(10, 30).tolist()})

    # last_time = 0
    # model = None
    # while model is None:
    #     model, last_time = load_model(last_time)
    #     time.sleep(0.1)
    # print(model, last_time)

    # data = []
    # while len(data) == 0:
    #     data, last_time = read_new_images(1670_139_800)
    #     time.sleep(0.1)
    # print("Last time:", last_time)

    # for (img, label) in data:
    #     print(img.shape)
    #     print(type(label))
