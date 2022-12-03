import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def run_model2(input):
    # See: https://tfhub.dev/tensorflow/collections/object_detection/1
    # See: https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2
    # See: https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb
    detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    input_sample = input[0, :, :, :][np.newaxis, :, :, :]
    detector_output = detector(input_sample)
    
    # print("detector variables:", detector_output.keys())
    # dict_keys(['raw_detection_scores', 'num_detections', 'detection_multiclass_scores',
    # 'detection_anchor_indices', 'detection_scores', 'detection_classes', 'detection_boxes', 'raw_detection_boxes'])

    print("class_ids: ", detector_output["detection_classes"])

def run_model(input, debug=False, model_path="model_ssd_mobilenet_v2_100_int8.tflite"):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path)

    # dynamic input size
    input_size = input[0, :, :, :][np.newaxis, :, :, :].shape
    print("input_size: ", input_size)

    interpreter.resize_tensor_input(0, input_size)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']

    if debug:
        input_sample = input[0, :, :, :][np.newaxis, :, :, :]
        input_sample = input_sample.view(input_details[0]['dtype'])
        print(input_sample.shape)
        interpreter.set_tensor(input_details[0]['index'], input_sample)

        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data.shape)
        print(output_data)
    else:
        for i in range(input.shape[0]):
            input_sample = input[i, :, :, :][np.newaxis, :, :, :]
            input_sample = input_sample.view(input_details[0]['dtype'])
            interpreter.set_tensor(input_details[0]['index'], input_sample)

            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            print(output_data.shape)
            print(output_data)        