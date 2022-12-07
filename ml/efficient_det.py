import tensorflow as tf
import tensorflow_hub as hub

import cv2

SHAPE = (320, 320)

img = cv2.imread("data/bus.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, SHAPE)
print(img.dtype, img.shape)

img_norm = (img / 255.0) * 2 - 1
img_norm = img_norm[None]
print(img_norm, img_norm.shape)

base_model = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1")
cls_outputs, box_outputs = base_model(img_norm, training=False)

print("classes")
print(cls_outputs)
print("boxes")
print(box_outputs)

