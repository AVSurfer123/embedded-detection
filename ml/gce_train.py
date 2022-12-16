import os
import sys
import subprocess
import time
from datetime import datetime
import numpy as np
import tensorflow as tf

sys.path.append('../automl/efficientdet/dataset')
sys.path.append('../networking/')

# from model_inspect import *
from data_utils import read_new_images, save_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_IMAGE_DIR = os.path.join(CURRENT_DIR, "../networking/server_images")
SERVER_LABEL_DIR = os.path.join(CURRENT_DIR, "../networking/server_labels")
TFLITE_FILEPATH = os.path.join(CURRENT_DIR, "../networking/server_weights/model.tflite")
LOOP_TIME = 0

print("Done loading imports. Launching training loop:")

start_read_time = time.time()
# while(True):

time.sleep(LOOP_TIME)
loop_start = datetime.now()
print("(1) Start Training Loop at: " + loop_start.strftime("%H:%M:%S"))

# Read data from disk ###################################################################
# print("(2) Read data w/ `read_new_images()` starting at: " \
#         + datetime.fromtimestamp(start_read_time).strftime('%H:%M:%S'))
# data, start_read_time = read_new_images(start_read_time, SERVER_IMAGE_DIR, SERVER_LABEL_DIR)

# print(len(data))
# print(data)
# data = [(np.array([5, 5, 3]), 2), (np.array([15, 15, 3]), 4), (np.array([25, 25, 3]), 6)]

# # Load Dataset ##########################################################################
# print("(3) Preproces and Load data")

# processed_data = list(zip(*data))
# print(processed_data)
# # images = tf.convert_to_tensor(processed_data[0])
# # annot = tf.convert_to_tensor(processed_data[1])
# images = processed_data[0]
# annot = processed_data[1]
# print(images, annot)

# # write images into tmp_data/images
# extension = '.png'
# for i in range(len(images)):
#     with open('tmp_data/images/' + str(i) + extension, 'w+') as f:
#         f.write(images[i])
#     f.close()

# # write annot.json into tmp_data/annot
# for i in range(len(annot)):
#     with open('tmp_data/annot/' + str(i) + extension, 'w+') as f:
#         f.write(annot[i])
#     f.close()


# subprocess.run(["python", "../automl/efficientdet/dataset/create_coco_tfrecord.py",
#                     "--image_dir=tmp_data/images",

#                     "--image_info_file=tmp_data/annot",
#                     "--object_annotations_file=tmp_data/annot",
                    
#                     "--output_file_prefix=tfrecord/coco"
#                 ])


print("SERVER_IMAGE_DIR: ", SERVER_IMAGE_DIR)
print("SERVER_LABEL_DIR: ", SERVER_LABEL_DIR)

subprocess.run(["python", "../automl/efficientdet/dataset/create_coco_tfrecord.py",
                    "--image_dir=" + SERVER_IMAGE_DIR,
                    "--object_annotations_file=" + os.path.join(SERVER_LABEL_DIR, "instances_retrain.json"),
                    "--output_file_prefix=tfrecord/coco"
                ])

# # Train, Save model chkpt ###############################################################
# print("(4) Run training from chkpt...")
# # train_file_pattern = glob-style path to training data, e.g. "fonts/*.csv"
# # val_file_pattern = glob-style path to val data, e.g. "fonts/*.csv"

# # train_input_fn = dataloader.InputReader(
# #   FLAGS.train_file_pattern,
# #   is_training=True,
# #   use_fake_data=FLAGS.use_fake_data,
# #   max_instances_per_image=max_instances_per_image)

# # eval_input_fn = dataloader.InputReader(
# #     FLAGS.val_file_pattern,
# #     is_training=False,
# #     use_fake_data=FLAGS.use_fake_data,
# #     max_instances_per_image=max_instances_per_image)

FILE_PATTERN = "tfrecord/*.tfrecord"
# # FILE_PATTERN = os.path.join(SERVER_IMAGE_DIR, "/*.txt")

subprocess.run(["python", "../automl/efficientdet/main.py",
                    "--mode=train_and_eval",
                    "--train_file_pattern=" + FILE_PATTERN,
                    "--val_file_pattern=" + FILE_PATTERN,
                    "--model_name=efficientdet-d0",
                    "--model_dir=/tmp/efficientdet-d0-finetune",
                    "--ckpt=efficientdet-d0",
                    "--train_batch_size=8",
                    "--eval_batch_size=8",
                    "--num_examples_per_epoch=10",
                    "--num_epochs=50",
                    "--hparams=coco_config.yaml"
                    # "--val_json_file=tfrecord/json_pascal.json"
                ])

# # Convert Model to .tflite and save #####################################################
# print("(5) Convert model to .tflite...")

# subprocess.run(["python", "../automl/efficientdet/model_inspect.py",
#                     "--runmode=saved_model",
#                     "--model_name=efficientdet-batch8",
#                     "--ckpt_path=../automl/efficientdet/efficientdet-batch8",
#                     "--saved_model_dir=batch8_savedmodeldir",
#                     "--tflite_path=" + TFLITE_FILEPATH,
#                     "--batch_size=8"
#                 ])

# loop_end = datetime.now()
# print("Training loop took ", loop_end - loop_start)
# print("Sleeping...")
