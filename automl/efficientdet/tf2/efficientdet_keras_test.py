# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for efficientdet_keras."""
import os
import tempfile
from absl import logging
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
import efficientdet_arch as legacy_arch
import hparams_config
from tf2 import efficientdet_keras
from tf2 import train_lib

SEED = 111111


class EfficientDetKerasTest(tf.test.TestCase):

  def test_irregular_shape(self):
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
    config.image_size = '896x1600'
    model = efficientdet_keras.EfficientDetNet(config=config)
    model(tf.ones([1, 896, 1600, 3]), False)
    model(tf.ones([1, 499, 333, 3]), False)

  def test_model_output(self):
    inputs_shape = [1, 512, 512, 3]
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
    config.heads = ['object_detection', 'segmentation']
    tf2.keras.utils.set_random_seed(SEED)
    with tf.Session(graph=tf.Graph()) as sess:
      feats = tf.ones(inputs_shape)
      model = efficientdet_keras.EfficientDetNet(config=config)
      outputs = model(feats, True)
      sess.run(tf.global_variables_initializer())
      keras_class_out, keras_box_out, _ = sess.run(outputs)
      grads = tf.nest.map_structure(lambda output: tf.gradients(output, feats),
                                    outputs)
      keras_class_grads, keras_box_grads, _ = sess.run(grads)
      vars = list(filter(
          lambda var: not var.name.startswith('segmentation'),
          tf.global_variables()))
      vars.sort(key=lambda var: var.name)
      keras_vars_names = [var.name for var in vars]
      keras_vars_values = sess.run(vars)

    with tf.Session(graph=tf.Graph()) as sess:
      feats = tf.ones(inputs_shape)
      outputs = legacy_arch.efficientdet(feats, config=config)
      vars = tf.global_variables()
      vars.sort(key=lambda var: var.name)
      legacy_vars_names = [var.name for var in vars]
      sess.run([var.assign(val) for val, var in zip(keras_vars_values, vars)])
      legacy_class_out, legacy_box_out = sess.run(outputs)
      grads = tf.nest.map_structure(lambda output: tf.gradients(output, feats),
                                    outputs)
      legacy_class_grads, legacy_box_grads = sess.run(grads)

    self.assertAllEqual(keras_vars_names, legacy_vars_names)

    for i in range(3, 8):
      self.assertAllEqual(
          keras_class_out[i - 3], legacy_class_out[i])
      self.assertAllEqual(
          keras_box_out[i - 3], legacy_box_out[i])
      self.assertAllEqual(
          keras_class_grads[i - 3], legacy_class_grads[i])
      self.assertAllEqual(
          keras_box_grads[i - 3], legacy_box_grads[i])

  def test_eager_output(self):
    inputs_shape = [1, 512, 512, 3]
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
    config.heads = ['object_detection']
    tmp_ckpt = os.path.join(tempfile.mkdtemp(), 'ckpt2')

    with tf.Session(graph=tf.Graph()) as sess:
      feats = tf.ones(inputs_shape)
      tf2.keras.utils.set_random_seed(SEED)
      model = efficientdet_keras.EfficientDetNet(config=config)
      outputs = model(feats, True)
      grads = tf.nest.map_structure(lambda output: tf.gradients(output, feats),
                                    outputs)
      sess.run(tf.global_variables_initializer())
      keras_class_out, keras_box_out = sess.run(outputs)
      legacy_class_grads, legacy_box_grads = sess.run(grads)
      model.save_weights(tmp_ckpt)

    feats = tf.ones(inputs_shape)
    model = efficientdet_keras.EfficientDetNet(config=config)
    model.build(inputs_shape)
    model.load_weights(tmp_ckpt)

    @tf.function
    def _run(feats):
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(feats)
        eager_class_out, eager_box_out = model(feats, True)
        class_grads, box_grads = tf.nest.map_structure(
                                      lambda output: tape.gradient(output, feats),
                                      [eager_class_out, eager_box_out])
      return eager_class_out, eager_box_out, class_grads, box_grads
    eager_class_out, eager_box_out, class_grads, box_grads = _run(feats)
    for i in range(5):
      self.assertAllEqual(
          eager_class_out[i], keras_class_out[i])
      self.assertAllEqual(
          eager_box_out[i], keras_box_out[i])
      self.assertAllEqual(
          class_grads[i], legacy_class_grads[i][0])
      self.assertAllEqual(
          box_grads[i], legacy_box_grads[i][0])

  def test_build_feature_network(self):
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
    config.max_level = 5
    with tf.Session(graph=tf.Graph()) as sess:
      inputs = [
          tf.ones([1, 64, 64, 40]),  # level 3
          tf.ones([1, 32, 32, 112]),  # level 4
          tf.ones([1, 16, 16, 320]),  # level 5
      ]
      tf2.keras.utils.set_random_seed(SEED)
      fpn_cell = efficientdet_keras.FPNCells(config)
      new_feats1 = fpn_cell(inputs, True)
      sess.run(tf.global_variables_initializer())
      keras_feats = sess.run(new_feats1)
      grads = tf.gradients(new_feats1, inputs)
      keras_grads = sess.run(grads)

    with tf.Session(graph=tf.Graph()) as sess:
      inputs = {
          0: tf.ones([1, 512, 512, 3]),
          1: tf.ones([1, 256, 256, 16]),
          2: tf.ones([1, 128, 128, 24]),
          3: tf.ones([1, 64, 64, 40]),
          4: tf.ones([1, 32, 32, 112]),
          5: tf.ones([1, 16, 16, 320])
      }
      tf2.keras.utils.set_random_seed(SEED)
      new_feats2 = legacy_arch.build_feature_network(inputs, config)
      sess.run(tf.global_variables_initializer())
      legacy_feats = sess.run(new_feats2)
      grads = tf.gradients(tf.nest.flatten(new_feats2), tf.nest.flatten(inputs))
      legacy_grads = sess.run(grads[3:6])

    for i in range(config.min_level, config.max_level + 1):
      self.assertAllEqual(keras_feats[i - config.min_level], legacy_feats[i])
      self.assertAllEqual(keras_grads[i - config.min_level],
                          legacy_grads[i - config.min_level])

  def test_model_variables(self):
    input_shape = (1, 512, 512, 3)
    model = efficientdet_keras.EfficientDetNet('efficientdet-d0')
    model.build(input_shape)
    eager_train_vars = sorted([var.name for var in model.trainable_variables])
    eager_model_vars = sorted([var.name for var in model.variables])
    with tf.Graph().as_default():
      feats = tf.ones([1, 512, 512, 3])
      model = efficientdet_keras.EfficientDetNet('efficientdet-d0')
      model(feats, True)
      keras_train_vars = sorted([var.name for var in tf.trainable_variables()])
      keras_model_vars = sorted([var.name for var in tf.global_variables()])
      keras_update_ops = [
          op.name for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      ]
    with tf.Graph().as_default():
      feats = tf.ones([1, 512, 512, 3])
      legacy_arch.efficientdet(feats, 'efficientdet-d0')
      legacy_train_vars = sorted([var.name for var in tf.trainable_variables()])
      legacy_model_vars = sorted([var.name for var in tf.global_variables()])
      legacy_update_ops = [
          op.name for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      ]
    self.assertEqual(keras_train_vars, legacy_train_vars)
    self.assertEqual(keras_model_vars, legacy_model_vars)
    self.assertEqual(eager_train_vars, legacy_train_vars)
    self.assertEqual(eager_model_vars, legacy_model_vars)
    self.assertAllEqual(keras_update_ops, legacy_update_ops)

  def test_resample_feature_map(self):
    feat = tf.random.uniform([1, 16, 16, 320])
    for apply_bn in [True, False]:
      for training in [True, False]:
        for strategy in ['tpu', '']:
          with self.subTest(
              apply_bn=apply_bn, training=training, strategy=strategy):
            tf2.keras.utils.set_random_seed(SEED)
            expect_result = legacy_arch.resample_feature_map(
                feat,
                name='resample_p0',
                target_height=8,
                target_width=8,
                target_num_channels=64,
                apply_bn=apply_bn,
                is_training=training,
                strategy=strategy)
            tf2.keras.utils.set_random_seed(SEED)
            resample_layer = efficientdet_keras.ResampleFeatureMap(
                name='resample_p0',
                feat_level=0,
                target_num_channels=64,
                apply_bn=apply_bn,
                is_training_bn=training,
                strategy=strategy)
            all_feats = [tf.ones([1, 8, 8, 64])]
            actual_result = resample_layer(feat, training, all_feats)
            self.assertAllCloseAccordingToType(expect_result, actual_result)

  def test_hub_model(self):
    image = tf.random.uniform((1, 320, 320, 3))
    keras_model = efficientdet_keras.EfficientDetNet('efficientdet-lite0')
    tmp_ckpt = os.path.join(tempfile.mkdtemp(), 'ckpt')
    keras_model.config.model_dir = tmp_ckpt
    base_model = train_lib.EfficientDetNetTrainHub(keras_model.config,
                                                   "https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1")
    cls_outputs, box_outputs = tf.function(base_model)(image, training=False)
    keras_model.build(image.shape)
    d1 = {var.name: var for var in base_model.variables}
    for var in keras_model.variables:
      var.assign(d1[var.name].numpy())
    cls_outputs2, box_outputs2 = tf.function(keras_model)(image, False)
    for c1, b1, c2, b2 in zip(cls_outputs, box_outputs, cls_outputs2, box_outputs2):
      self.assertAllEqual(c1, c2)
      self.assertAllEqual(b1, b2)

  def test_resample_var_names(self):
    with tf.Graph().as_default():
      feat = tf.random.uniform([1, 16, 16, 320])
      resample_layer = efficientdet_keras.ResampleFeatureMap(
          name='resample_p0',
          feat_level=0,
          target_num_channels=64)
      all_feats = [tf.ones([1, 8, 8, 64])]
      resample_layer(feat, True, all_feats)
      vars1 = sorted([var.name for var in tf.trainable_variables()])

    with tf.Graph().as_default():
      feat = tf.random.uniform([1, 16, 16, 320])
      legacy_arch.resample_feature_map(
          feat,
          name='p0',
          target_height=8,
          target_width=8,
          target_num_channels=64)
      vars2 = sorted([var.name for var in tf.trainable_variables()])

    self.assertEqual(vars1, vars2)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
