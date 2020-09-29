from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
import os
import subprocess

import random

import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

mape = tf.keras.losses.MeanAbsolutePercentageError()
mse = tf.keras.losses.MeanSquaredError()

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


WORKING_DIR = os.getcwd()


def download_files_from_gcs(source, destination):
  """Download files from GCS to a WORKING_DIR/.

  Args:
    source: GCS path to the training data
    destination: GCS path to the validation data.

  Returns:
    A list to the local data paths where the data is downloaded.
  """
  local_file_names = [destination]
  gcs_input_paths = [source]

  # Copy raw files from GCS into local path.
  raw_local_files_data_paths = [os.path.join(WORKING_DIR, local_file_name)
    for local_file_name in local_file_names
    ]
  for i, gcs_input_path in enumerate(gcs_input_paths):
    if gcs_input_path:
      subprocess.check_call(
        ['gsutil', 'cp', gcs_input_path, raw_local_files_data_paths[i]])

  return raw_local_files_data_paths


def _load_data(path, destination):
  """Verifies if file is in Google Cloud.

  Args:
    path: (str) The GCS URL to download from (e.g. 'gs://bucket/file.csv')
    destination: (str) The filename to save as on local disk.

  Returns:
    A filename
  """
  if path.startswith('gs://'):
    download_files_from_gcs(path, destination=destination)
    return destination
  return path


def generator(data, lookback, delay=0,
              shuffle=False, batch_size=128, step=1, split_index=None):
    max_index = len(data) - delay - 1
    if split_index is not None:
        split_index = split_index - delay - 1
    i = lookback
    while 1:
        if shuffle:
            rows = np.random.randint(lookback, max_index, size=batch_size)
        else:
            if split_index is not None:
                if i + batch_size >= split_index:
                    i = split_index + lookback
                elif i + batch_size >= max_index:
                    i = lookback

                if i >= split_index:
                    rows = np.arange(i, min(i + batch_size, max_index))
                else:
                    rows = np.arange(i, min(i + batch_size, split_index))

            else:
                if i + batch_size >= max_index:
                    i = lookback
                rows = np.arange(i, min(i + batch_size, max_index))

            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay, 0]
        yield samples, targets


def to_savedmodel(model, export_path):
  """Convert the Keras HDF5 model into TensorFlow SavedModel."""

  builder = saved_model_builder.SavedModelBuilder(export_path)

  signature = predict_signature_def(
      inputs={'input': model.inputs[0]}, outputs={'income': model.outputs[0]})

  with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
        })
    builder.save()

def mse_custom(y_true, y_pred):
    return mse(y_true, y_pred)

def mape_custom(y_true, y_pred):
    return mape(y_true, y_pred)


def u_theil(y_true, y_pred):

    if K.int_shape(y_true)[0] is None:
        return 0.0
    else:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        error_sup = K.sum(K.square(y_true - y_pred))
        error_inf = K.sum(K.square(y_pred[0:(K.int_shape(y_pred)[0] - 1)] - y_pred[1:K.int_shape(y_pred)[0]]))

        return error_sup / error_inf


def prediction_of_change_in_direction(y_true, y_pred):

    if K.int_shape(y_true)[0] is None:
        return 50.0
    else:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        true_sub = y_true[0:(K.int_shape(y_true)[0] - 1)] - y_true[1:K.int_shape(y_true)[0]]
        pred_sub = y_pred[0:(K.int_shape(y_pred)[0] - 1)] - y_pred[1:K.int_shape(y_pred)[0]]

        mult = true_sub * pred_sub

        result = K.sum(K.clip(K.sign(mult), 0.0, 1), axis=0)

        return (100 * (result / (K.int_shape(y_true)[0]-1)))


def prediction_of_change_in_direction_ret(y_true, y_pred):

    if K.int_shape(y_true)[0] is None:
        return 50.0
    else:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mult = y_true * y_pred

        result = K.sum(K.clip(K.sign(mult), 0.0, 1), axis=0)

        return (100 * (result / K.int_shape(y_true)[0]))


def average_relative_variance(y_true, y_pred):

    if K.int_shape(y_true)[0] is None:
        return 1.0
    else:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mean = K.mean(y_true)

        error_sup = K.sum(K.square(y_true - y_pred))
        error_inf = K.sum(K.square(y_pred - K.ones(K.int_shape(y_pred)[0]) * mean))

        return error_sup / error_inf


def index_agreement(y_true, y_pred):

    if K.int_shape(y_true)[0] is None:
        return 0.0
    else:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mean = K.mean(y_true)

        error_sup = K.sum(K.square(K.abs(y_true - y_pred)))

        error_inf = K.abs(y_pred - K.ones(K.int_shape(y_pred)[0]) * mean) + K.abs(y_true - K.ones(K.int_shape(y_true)[0]) * mean)
        error_inf = K.sum(K.square(error_inf))

        return (1 - (error_sup / error_inf))


def sum_of_losses_and_gains(y_true, y_pred):

    if K.int_shape(y_true)[0] is None:
        return 0.0
    else:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        true_sub = y_true[0:(K.int_shape(y_true)[0] - 1)] - y_true[1:K.int_shape(y_true)[0]]
        pred_sub = y_pred[0:(K.int_shape(y_pred)[0] - 1)] - y_pred[1:K.int_shape(y_pred)[0]]

        mult = true_sub * pred_sub

        result = K.sum(K.abs(y_true[0:(K.int_shape(y_true)[0] - 1)] - y_true[1:K.int_shape(y_true)[0]]) * K.sign(mult), axis=0)

        return (result / (K.int_shape(y_true)[0]-1))


def sum_of_losses_and_gains_ret(y_true, y_pred, sample_weight=None):

    if K.int_shape(y_true)[0] is None:
        return 0.0
    else:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mult = y_true * y_pred

        result = K.sum(K.abs(y_true) * K.sign(mult), axis=0)

        if sample_weight is not None:
            result = result * sample_weight

        return (result / K.int_shape(y_true)[0])

