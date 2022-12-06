import numpy as np

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.lite as tflite

# # Works, but fp32 and needs different dataset to do int quantization
# m = tf.keras.Sequential([
#     hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
#                    trainable=True),  # Can be True, see below.
#     tf.keras.layers.Dense(91, activation='softmax')
# ])

IMAGE_SHAPE = (224, 224, 3)
# # inputs = tf.keras.layers.Input(shape=(320, 320, 3))
# # base = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1")

# # m2 = tf.keras.Sequential([base])
# # m2.build([None, 320, 320, 3])

# cls = tf.keras.layers.Dense(91, name="class")(m2.output)
# # bb = tf.keras.layers.Dense(91, name="bb")(base.output)

# # m = tf.keras.Sequential([base, [p1, d1]])
# m = tf.Model(base.input, outputs=[cls, bb])

# grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
# grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
# grace_hopper = np.array(grace_hopper)/255.0
# result = m.predict(grace_hopper[np.newaxis, ...])
# print(result)

# # m.build([None, 320, 320, 3])  # Batch input shape.
# # m.summary()

##################################################################################


base_model = tf.keras.applications.efficientnet.EfficientNetB0(input_shape = IMAGE_SHAPE, include_top=True, weights='imagenet')
base_model.trainable = False
base_model.summary()

converter = tflite.TFLiteConverter.from_keras_model(base_model)



##################################################################################

# # Everything below "works" except for 'representative_dataset'
# # Save model 
# tf.saved_model.save(m, 'model_test_save')

# # def representative_dataset():
# #   for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
# #     yield [input_value]

# # Convert to tflite
# converter = tflite.TFLiteConverter.from_keras_model(m)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# # converter.representative_dataset = representative_dataset
# # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# # converter.inference_input_type = tf.int8  # or tf.uint8
# # converter.inference_output_type = tf.int8  # or tf.uint8

# tflite_model = converter.convert()

# model_path = 'test99.tflite'
# with open(model_path, 'wb') as f:
#     f.write(tflite_model)
# f.close()

# # # Load TFLite model using interpreter and allocate tensors.
# # interpreter = tflite.Interpreter(model_path=model_path)
# # interpreter.allocate_tensors()