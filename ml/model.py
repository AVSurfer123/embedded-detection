import numpy as np
import tensorflow as tf

def run_model(input, model_path="model_ssd_mobilenet_v2_100_int8.tflite"):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path)

    # Get input and output tensors.
    interpreter.resize_tensor_input(0, [1, 720, 1280, 3])
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    print("input_shape:", input_shape)
    print(input_details[0]['dtype'])

    # input_sample = np.array(np.random.random_sample(input_shape), dtype=np.int8)
    # for i in range(input.shape[0]):
    i = 0

    input_sample = input[i, :, :, :]
    input_sample = input_sample[np.newaxis, :, :, :]
    
    print(input_sample.dtype)
    print(input_sample.shape)
    print(input_sample)

    input_sample = input_sample.view(input_details[0]['dtype'])

    print(input_sample.dtype)
    print(input_sample.shape)
    print(input_sample)
    # # input_sample = np.array(np.random.randint(-128, 127, input_shape, dtype=np.int8))
    # # print("random input data: ", input_sample)
    interpreter.set_tensor(input_details[0]['index'], input_sample)

    interpreter.invoke()

    # # The function `get_tensor()` returns a copy of the tensor data.
    # # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)
    print(output_data)
