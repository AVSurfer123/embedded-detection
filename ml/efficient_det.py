import tensorflow as tf
import tensorflow_hub as hub

import cv2

def gen_tflite():
    # SHAPE = (320, 320)

    # img = cv2.imread("data/bus.jpg", cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, SHAPE)
    # print("image:", img.dtype, img.shape)

    # img_norm = (img / 255.0) * 2 - 1
    # img_norm = img_norm[None]
    # print(img_norm, img_norm.shape)

    # base_model = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1")
    # cls_outputs, box_outputs = base_model(img_norm, training=False)

    # print(len(cls_outputs))
    # for i in cls_outputs:
    #     print(i.shape)

    # print(len(box_outputs))
    # for j in box_outputs:
    #     print(j.shape)
    
    input = tf.keras.Input(shape=(None, 320, 320, 3))
    base_model = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1")
    x = base_model(input)

    model = tf.keras.Model(inputs=input, outputs=x, name="temp_model")
    tf.saved_model.save(model, ".")

    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.representative_dataset = representative_dataset
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.int8  # or tf.uint8
    # converter.inference_output_type = tf.int8  # or tf.uint8
    # tflite_model = converter.convert()

    # Save the model.
    with open('z_model_new.tflite', 'wb') as f:
        f.write(tflite_model)

def from_saved():
    converter = tf.lite.TFLiteConverter.from_saved_model('z_model_new.tflite')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    with open('z_model_new2.tflite', 'wb') as f:
        f.write(tflite_quant_model)

if __name__ == "__main__":
    from_saved()