import sys
sys.path.append('../automl/efficientdet/')
sys.path.append('../networking/')

# import os
# import time
# from typing import Text, Tuple, List
# from absl import app
# from absl import flags
# from absl import logging
# import numpy as np
# from PIL import Image
# import tensorflow.compat.v1 as tf
# import hparams_config
# import inference
# import utils
# from tensorflow.python.client import timeline

from model_inspect import *
from data_utils import read_new_images, save_model
import subprocess
import time
from datetime import datetime


# print("hello")
# app.run(main)

LOOP_TIME = 5

print("Done loading imports. Launching training loop:")

while(True):
    start_read_time = time.time()
    time.sleep(LOOP_TIME)

    print("Start Training Loop at " + datetime.now().strftime("%H:%M:%S"))
    print("Reading data starting at " + datetime.fromtimestamp(start_read_time).strftime('%H:%M:%S'))
    read_new_images(start_read_time)

    #
    # model from disk
    # tf_dataloader
    # train
    # store/save model
    # convert
    # save_model(latest_tflite_model)
    print("Sleeping...")
    break