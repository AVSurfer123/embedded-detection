"""
Run object detection on images, Press ESC to exit the program
For Raspberry PI, please use `import tflite_runtime.interpreter as tflite` instead
"""
import re
import cv2
import numpy as np
from datetime import datetime

import tensorflow.lite as tflite
import tensorflow_hub as hub
import tensorflow as tf

from PIL import Image


def load_labels(label_path):
    r"""Returns a list of labels"""
    with open(label_path) as f:
        labels = {}
        for line in f.readlines():
            m = re.match(r"(\d+)\s+(\w+)", line.strip())
            labels[int(m.group(1))] = m.group(2)
        return labels


def load_model(model_path):
    r"""Load TFLite model, returns a Interpreter instance."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def process_image(interpreter, image, input_index):
    r"""Process an image, Return a list of detected class ids and positions"""
    input_data = np.expand_dims(image, axis=0)  # expand to 4-dim

    # Process
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Get outputs
    output_details = interpreter.get_output_details()
    # print(output_details)
    # output_details[0] - position
    # output_details[1] - class id
    # output_details[2] - score
    # output_details[3] - count
    
    positions = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))

    # print(positions)
    # print(classes)
    # print(scores)

    result = []

    for idx, score in enumerate(scores):
        if score > 0.5:
            result.append({'pos': positions[idx], '_id': classes[idx] })

    return result

def process_image2(classifier, image, input_index):
    r"""Process an image, Return a list of detected class ids and positions"""

    input_data = np.expand_dims(image, axis=0)  # expand to 4-dim
    result = classifier.predict(input_data)
    print(result.shape)

    # # Process
    # interpreter.set_tensor(input_index, input_data)
    # interpreter.invoke()

    # # Get outputs
    # output_details = interpreter.get_output_details()
    # # print(output_details)
    # # output_details[0] - position
    # # output_details[1] - class id
    # # output_details[2] - score
    # # output_details[3] - count
    
    # positions = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    # classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    # scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))

    # # print(positions)
    # # print(classes)
    # # print(scores)

    # result = []

    # for idx, score in enumerate(scores):
    #     if score > 0.5:
    #         result.append({'pos': positions[idx], '_id': classes[idx] })

    return result

def display_result(result, frame, labels):
    r"""Display Detected Objects"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.6
    color = (255, 0, 0)  # Blue color
    thickness = 1

    # position = [ymin, xmin, ymax, xmax]
    # x * IMAGE_WIDTH
    # y * IMAGE_HEIGHT
    width = frame.shape[1]
    height = frame.shape[0]

    for obj in result:
        pos = obj['pos']
        _id = obj['_id']

        x1 = int(pos[1] * width)
        x2 = int(pos[3] * width)
        y1 = int(pos[0] * height)
        y2 = int(pos[2] * height)

        cv2.putText(frame, labels[_id], (x1, y1), font, size, color, thickness)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow('Object Detection', frame)


if __name__ == "__main__":

    model_path = 'data/detect.tflite'
    # model_path = 'test99.tflite'
    label_path = 'data/coco_labels.txt'
    image_path = 'data/bus.jpg'
    write_path = 'pred/'

    interpreter = load_model(model_path)
    labels = load_labels(label_path)

    # classifier_model = "https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1"
    # classifier = tf.keras.Sequential([
    #     hub.KerasLayer(classifier_model, input_shape=(320, 320)+(3,))
    # ])

    # base_model = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1")

    input_details = interpreter.get_input_details()
    # Get Width and Height
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]

    # Get input index
    input_index = input_details[0]['index']

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # width = 320
    # height = 320
    image = image.resize((width, height))

    # top_result = dict of {pos: [pos coord %], id: class_id} bounding boxes where score > 0.5
    top_result = process_image(interpreter, image, input_index)
    # top_result = process_image2(classifier, image, input_index)
    # print(top_result)

    # timestamp = datetime.now().strftime("%a%d_%H_%M_%S")
    # filepath = write_path + "single_image/" + timestamp
    # with open(filepath, 'w') as f:
    #     np.save(filepath, top_result)
    # f.close()

    display_result(top_result, frame, labels)

    key = cv2.waitKey(0)
    if key == 27:  # esc
        cv2.destroyAllWindows()
