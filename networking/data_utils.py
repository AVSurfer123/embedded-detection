import numpy as np
from skimage import io
from tensorflow import keras
import os
import time
import json
from typing import List, Tuple, Any

IMAGE_DIR = "images"
LABEL_DIR = "labels"
WEIGHT_DIR = "weights"

SERVER_IMAGE_DIR = "server_images"
SERVER_LABEL_DIR = "server_labels"
SERVER_WEIGHT_DIR = "server_weights"

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
    for file_name in os.listdir(SERVER_IMAGE_DIR):
        file_path = os.path.join(SERVER_IMAGE_DIR, file_name)
        # mtime = os.stat(file_path).st_mtime
        mtime = int(file_name.split('.')[0]) / 1000 
        if mtime > last_time:
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

def load_weights(model: keras.Model, last_time: float) -> Tuple[bool, float]:
    """
    Client method: Loads weights from disk into the model, if newer weights have arrived from the central server.

    Args:
        model: Keras model to load weights into
        last_time: Only loads the weights if they arrived after this time. Represented as seconds after Unix epoch, like time.time()

    Returns:
        Tuple of
            Success for whether new weights were loaded or no new weights were found,
            New last_time to use which is modification time of model on the disk
    """
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    file_name = os.path.join(WEIGHT_DIR, "model.h5")
    mtime = os.stat(file_name).st_mtime
    if mtime > last_time:
        model.load_weights(os.path.join(WEIGHT_DIR, "model.h5"))
        return True, mtime
    return False, last_time

def save_weights(model: keras.Model):
    """
    Server method: Saves the existing weights from the model to the disk to send to the edge clients.

    Args:
        model: Keras model to save
    """
    os.makedirs(SERVER_WEIGHT_DIR, exist_ok=True)
    model.save_weights(os.path.join(SERVER_WEIGHT_DIR, "model.h5"))


if __name__ == '__main__':
    print('hello i started')
    image = io.imread("/home/ubuntu/purple_tree.jpg")
    # image.show()
    write_image(image, {'testing': 'the json', 'of': 123, 'dreams': np.random.rand(10, 30).tolist()})

    # data, last_time = read_new_images(1669250000)
    # print("Last time:", last_time)

    # for (img, label) in data:
    #     print(img.shape)
    #     print(type(label))
