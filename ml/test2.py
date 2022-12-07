import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.lite as tflite

detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1")

# Everything below "works" except for 'representative_dataset'
# Save model 
tf.saved_model.save(detector, 'z_test')

cifar = tf.keras.datasets.cifar100
(train_images, train_labels), (test_images, test_labels) = cifar.load_data()

def representative_dataset():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    yield [input_value]

# Convert to tflite
converter = tflite.TFLiteConverter.from_keras_model(detector)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8

tflite_model = converter.convert()

model_path = 'z_test99.tflite'
with open(model_path, 'wb') as f:
    f.write(tflite_model)
f.close()

# # Load TFLite model using interpreter and allocate tensors.
# interpreter = tflite.Interpreter(model_path=model_path)
# interpreter.allocate_tensors()