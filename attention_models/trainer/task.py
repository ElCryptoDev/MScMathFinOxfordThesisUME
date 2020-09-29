from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from datetime import datetime
import yaml
import pandas as pd
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.lib.io import file_io
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K




import io
# tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_eager_execution()
config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

from . import model
from . import darnn_model
from . import self_attn_model
from . import utils
from .utils import prediction_of_change_in_direction, prediction_of_change_in_direction_ret, u_theil, \
    average_relative_variance, index_agreement, \
    sum_of_losses_and_gains, sum_of_losses_and_gains_ret, mse_custom, mape_custom


def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='GCS location to write checkpoints and export models')

    parser.add_argument(
        '--run-name',
        type=str,
        required=True,
        help='Name of experiment')

    parser.add_argument(
        '--coin',
        type=str,
        default='BTC',
        choices=['BTC', 'ETH', 'XMR', 'XRP', 'LTC', 'BCH', 'XLM', 'DASH', 'IOTA', 'TRX', 'ndx', 'DJIA'],
        help='Name of coin to perfrom training for.')

    parser.add_argument(
        '--run_config',
        type=str,
        default=r"C:\Users\admsassl\PycharmProjects\CryptoMA\TradingBot\attention_models\run_config.yaml",
        help='Path of yaml file for configuration of target and features')

    parser.add_argument(
        '--sample',
        choices=['1m', '10m', '1h', '1d'],
        default='10m',
        help='Sample type to load. Available frequencies "10m", "1h" and "1d"')

    parser.add_argument(
        '--data-path',
        type=str,
        default="C:\\Users\\admsassl\\PycharmProjects\\CryptoMA\\Samples",
        help='Location of training files (local or GCS e.g. gs://sample_ai_bucket/)')

    parser.add_argument(
        '--loss',
        default='mean_squared_error',
        type=str,
        choices=['mean_squared_error', 'mean_absolute_percentage_error', 'prediction_of_change_in_direction',
                 'prediction_of_change_in_direction_ret', 'u_theil',
                 'average_relative_variance', 'index_agreement', 'sum_of_losses_and_gains',
                 'sum_of_losses_and_gains_ret', 'binary_crossentropy'],
        help='Loss function for training')

    parser.add_argument(
        '--num-layers',
        default=2,
        type=int,
        help='number of layers in LSTM model, default=2')

    parser.add_argument(
        '--num-nodes',
        default=128,
        type=int,
        help='number of layers in LSTM model, default=128')

    parser.add_argument(
        '--frac-dropout',
        type=float,
        default=0.0,
        help='dropout fraction, default=0')

    parser.add_argument(
        '--num-epochs',
        type=int,
        default=5,
        help='number of times to go through the data, default=5')

    parser.add_argument(
        '--batch-size',
        default=512,
        type=int,
        help='number of records to read during each training step, default=512')

    parser.add_argument(
        '--lookback',
        default=60,
        type=int,
        help='number of lookback time steps  in input sequence, default=60')

    parser.add_argument(
        '--step',
        default=1,
        type=int,
        help='step size for selection timesteps in input sequance, default=1')

    parser.add_argument(
        '--delay',
        default=0,
        type=int,
        help='time difference of input series\' end and prediction timestep, default=0 (1 time step ahead)')

    parser.add_argument(
        '--steps-per-epoch-override',
        type=int,
        help='override value for number of steps per epoch')

    parser.add_argument(
        '--temporal-attention',
        type=bool,
        default=False,
        help='Add temporal attention layer on bottom of LSTM network.')

    parser.add_argument(
        '--input-attention',
        type=bool,
        default=False,
        help='Add input attention layer on top of LSTM network.')

    parser.add_argument(
        '--attention-style',
        default='luong',
        choices=['luong', 'bahdanau'],
        help='Configure attention stlye: \'luong\' or \'bahdanau\')')

    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')

    parser.add_argument(
        '--validation-freq',
        default=1,
        type=int,
        help='frequency to carry out validation steps in times of epochs.')

    parser.add_argument(
        '--model_type',
        default='simple',
        type=str,
        choices=['simple', 'darnn', 'self_attn'],
        help='Selector for model architecture.')

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for Adam optimizer.')

    parser.add_argument(
        '--run-config',
        type=str,
        help='Location of yaml parameter configuration file.')

    parser.add_argument(
        '--categorical',
        type=bool,
        default=False,
        help='Type of prediction: categorical vs. regression.')

    return parser.parse_args()


# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def train_and_evaluate(hparams):
    """Helper function: Trains and evaluates model.

    Args:
    hparams: (dict) Command line parameters passed from task.py
    """
    try:
        os.makedirs(hparams['job_dir'])
    except:
        pass
    try:
        os.makedirs(os.path.join(hparams['job_dir'], "logs"))
    except:
        pass

    # Read config yaml (may override command line arguments)
    if hparams['run_config'] is not None:
        path_run_config = hparams['run_config']
        if hparams['run_config'].startswith('gs://'):
            path_run_config = utils._load_data(hparams['run_config'], 'run_config.yaml')
        with open(path_run_config) as file:
            dict_config = yaml.safe_load(file)
        for key, value in dict_config.items():
            hparams[key] = value

    hparams['run_name'] = hparams['run_name'] + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")
    MODEL_NAME = hparams['run_name']
    os.mkdir(os.path.join(hparams['job_dir'], MODEL_NAME))
    model_folder = os.path.join(hparams['job_dir'], MODEL_NAME)

    CHECKPOINT_FILE_NAME = 'checkpoint.{epoch:02d}.ckpt'
    checkpoint_path = os.path.join(model_folder, CHECKPOINT_FILE_NAME)
    logdir = os.path.join(hparams['job_dir'], "logs", MODEL_NAME)

    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    model_json_name = MODEL_NAME + ".json"

    path_train = hparams['data_path'] + '/sample_' + hparams['coin'] + '_' + hparams['sample'] + '_train.csv'
    path_val = hparams['data_path'] + '/sample_' + hparams['coin'] + '_' + hparams['sample'] + '_val.csv'
    path_test = hparams['data_path'] + '/sample_' + hparams['coin'] + '_' + hparams['sample'] + '_test.csv'

    if hparams['data_path'].startswith('gs://'):
        path_train = utils._load_data(path_train, 'train.csv')
        path_val = utils._load_data(path_val, 'val.csv')
        path_test = utils._load_data(path_val, 'test.csv')

    df_train = pd.read_csv(path_train)
    df_val = pd.read_csv(path_val)
    df_test = pd.read_csv(path_test)

    split_index = None
    if hparams['coin'] not in ['ndx', 'DJIA']:
        # Dirty hack to make train set bigger
        df_train_len = len(df_train.index)
        df_val_len = len(df_val.index)
        split_val = int((df_train_len + df_val_len) * 0.8 - df_train_len)
        df_train = pd.concat([df_train, df_val[:split_val]])
        df_val = df_val[split_val:]
        split_index = df_train_len

    path_target_features = hparams['target_features']
    if hparams['target_features'].startswith('gs://'):
        path_target_features = utils._load_data(hparams['target_features'], 'target_features.yaml')

    with open(path_target_features) as file:
        dict_target_features = yaml.safe_load(file)

    hparams['target'] = dict_target_features["target"]

    # Control structur for categorical prediction
    if hparams['categorical']:
        for df in [df_train, df_val, df_test]:
            df.loc[df[hparams['target']] >= 0.0, 'LABEL'] = 1
            df.loc[df[hparams['target']] < 0.0, 'LABEL'] = 0
            df['LABEL'] = df['LABEL'].astype(int)
        hparams['target'] = 'LABEL'

    SELECTED_COLS = dict_target_features["features"]
    try:
        SELECTED_COLS.remove(hparams['target'])
    except:
        pass
    #for i in SELECTED_COLS:
    #    hparams[i] = 'Y'
    SELECTED_COLS = [hparams['target']] + SELECTED_COLS

    # Mean-Var scaler (estimation based only on train set)
    mean = df_train.loc[:, SELECTED_COLS[1:]].mean(axis=0)
    df_train[SELECTED_COLS[1:]] -= mean
    df_val[SELECTED_COLS[1:]] -= mean
    df_test[SELECTED_COLS[1:]] -= mean
    std = df_train.loc[:, SELECTED_COLS[1:]].std(axis=0)
    # Remove all columns with zero variance
    SELECTED_COLS = [hparams['target']] + std[std != 0.0].index.tolist()
    df_train[SELECTED_COLS[1:]] /= std
    df_val[SELECTED_COLS[1:]] /= std
    df_test[SELECTED_COLS[1:]] /= std

    lookback = hparams['lookback']
    step = hparams['step']
    delay = hparams['delay']
    batch_size = hparams['batch_size']

    train_gen = utils.generator(df_train[SELECTED_COLS].values,
                                lookback=lookback,
                                delay=delay,
                                step=step,
                                batch_size=batch_size,
                                split_index=split_index)
    val_gen = utils.generator(df_val[SELECTED_COLS].values,
                              lookback=lookback,
                              delay=delay,
                              step=step,
                              batch_size=batch_size)
    test_gen = utils.generator(df_test[SELECTED_COLS].values,
                              lookback=lookback,
                              delay=delay,
                              step=step,
                              batch_size=batch_size)

    train_steps = (len(df_train) - lookback) // batch_size
    val_steps = (len(df_val) - lookback) // batch_size
    test_steps = (len(df_test) - lookback) // batch_size

    train_samples, train_targets = next(train_gen)
    for i in range(1, train_steps):
        samples_it, targets_it = next(train_gen)
        train_samples = np.concatenate([train_samples, samples_it], axis=0)
        train_targets = np.concatenate([train_targets, targets_it], axis=0)

    val_samples, val_targets = next(val_gen)
    for i in range(1, val_steps):
        samples_it, targets_it = next(val_gen)
        val_samples = np.concatenate([val_samples, samples_it], axis=0)
        val_targets = np.concatenate([val_targets, targets_it], axis=0)

    test_samples, test_targets = next(test_gen)
    for i in range(1, test_steps):
        samples_it, targets_it = next(test_gen)
        test_samples = np.concatenate([test_samples, samples_it], axis=0)
        test_targets = np.concatenate([test_targets, targets_it], axis=0)

    if hparams["steps_per_epoch_override"] is not None:
        hparams['steps_per_epoch'] = int(hparams['steps_per_epoch_override'])
    else:
        hparams['steps_per_epoch'] = int(train_steps)
    hparams.pop('steps_per_epoch_override', None)

    preds = train_samples[:, -1, 1]
    hparams['Persistence_mae_train'] = float(mse_custom(preds, train_targets))
    hparams['Persistence_mape_train'] = float(mape_custom(preds, train_targets))
    hparams['Persistence_POCID_train'] = float(prediction_of_change_in_direction(preds, train_targets))
    hparams['Persistence_POCID_ret_train'] = float(prediction_of_change_in_direction_ret(preds, train_targets))
    hparams['Persistence_U_THEIL_train'] = float(u_theil(preds, train_targets))
    hparams['Persistence_ARV_train'] = float(average_relative_variance(preds, train_targets))
    hparams['Persistence_IA_train'] = float(index_agreement(preds, train_targets))
    hparams['Persistence_SLG_train'] = float(sum_of_losses_and_gains(preds, train_targets))
    hparams['Persistence_SLG_ret_train'] = float(sum_of_losses_and_gains_ret(preds, train_targets))

    preds = val_samples[:, -1, 1]
    hparams['Persistence_mae_val'] = float(mse_custom(preds, val_targets))
    hparams['Persistence_mape_val'] = float(mape_custom(preds, val_targets))
    hparams['Persistence_POCID_val'] = float(prediction_of_change_in_direction(preds, val_targets))
    hparams['Persistence_POCID_ret_val'] = float(prediction_of_change_in_direction_ret(preds, val_targets))
    hparams['Persistence_U_THEIL_val'] = float(u_theil(preds, val_targets))
    hparams['Persistence_ARV_val'] = float(average_relative_variance(preds, val_targets))
    hparams['Persistence_IA_val'] = float(index_agreement(preds, val_targets))
    hparams['Persistence_SLG_val'] = float(sum_of_losses_and_gains(preds, val_targets))
    hparams['Persistence_SLG_ret_val'] = float(sum_of_losses_and_gains_ret(preds, val_targets))

    preds = test_samples[:, -1, 1]
    hparams['Persistence_mae_test'] = float(mse_custom(preds, test_targets))
    hparams['Persistence_mape_test'] = float(mape_custom(preds, test_targets))
    hparams['Persistence_POCID_test'] = float(prediction_of_change_in_direction(preds, test_targets))
    hparams['Persistence_POCID_ret_test'] = float(prediction_of_change_in_direction_ret(preds, test_targets))
    hparams['Persistence_U_THEIL_test'] = float(u_theil(preds, test_targets))
    hparams['Persistence_ARV_test'] = float(average_relative_variance(preds, test_targets))
    hparams['Persistence_IA_test'] = float(index_agreement(preds, test_targets))
    hparams['Persistence_SLG_test'] = float(sum_of_losses_and_gains(preds, test_targets))
    hparams['Persistence_SLG_ret_test'] = float(sum_of_losses_and_gains_ret(preds, test_targets))

    if hparams['temporal_attention'] is None:
        hparams['temporal_attention'] = False
    if hparams['frac_dropout'] == 0.0:
        hparams['frac_dropout'] = False

    if hparams['model_type'] == 'simple':
        mlmodel = model.value_rnn_model(num_values=len(SELECTED_COLS),
                                        lookback=lookback,
                                        num_layers=hparams['num_layers'],
                                        num_nodes=hparams['num_nodes'],
                                        dropout=hparams['frac_dropout'],
                                        input_attention=hparams['input_attention'],
                                        temporal_attention=hparams['temporal_attention'],
                                        attention_stlye=hparams['attention_style'],
                                        categorical=hparams['categorical'])
    elif hparams['model_type'] == 'darnn':
        mlmodel = darnn_model.darnn_model(num_values=len(SELECTED_COLS),
                                          num_nodes=hparams['num_nodes'],
                                          lookback=hparams['lookback'],
                                          input_attention=hparams['input_attention'],
                                          temporal_attention=hparams['temporal_attention'],
                                          categorical=hparams['categorical'])
        train_samples = [train_samples, train_samples[:, :, -1]]
        val_samples = [val_samples, val_samples[:, :, -1]]
        test_samples = [test_samples, test_samples[:, :, -1]]
    elif hparams['model_type'] == 'self_attn':
        if 'full_enc_pred' in hparams.keys():
            full_enc_pred = hparams['full_enc_pred']
            print('full_enc_pred: ' + str(full_enc_pred))
        else:
            full_enc_pred = True
        mlmodel = self_attn_model.self_attn_model(num_values=len(SELECTED_COLS),
                                                  d_model=hparams['num_nodes'],
                                                  num_heads=hparams['num_layers'],
                                                  lookback=hparams['lookback'],
                                                  batch_size=hparams['batch_size'],
                                                  full_enc_pred=full_enc_pred,
                                                  categorical=hparams['categorical'])

    hparams['trainable_parameters'] = float(np.sum([K.count_params(w) for w in mlmodel.trainable_weights]))

    optimizer = Adam(lr=hparams['learning_rate'])
    if hparams['loss'] == 'prediction_of_change_in_direction':
        loss = prediction_of_change_in_direction
    elif hparams['loss'] == 'u_theil':
        loss = u_theil
    elif hparams['loss'] == 'average_relative_variance':
        loss = average_relative_variance
    elif hparams['loss'] == 'index_agreement':
        loss = index_agreement
    elif hparams['loss'] == 'sum_of_losses_and_gains':
        loss = sum_of_losses_and_gains
    elif hparams['loss'] == 'sum_of_losses_and_gains_ret':
        loss = sum_of_losses_and_gains_ret
    elif hparams['loss'] == 'prediction_of_change_in_direction_ret':
        loss = prediction_of_change_in_direction_ret
    elif hparams['loss'] == 'binary_crossentropy':
        loss = 'binary_crossentropy'
    else:
        loss = hparams['loss']

    if hparams['categorical']:
        metrics = ['accuracy']
    else:
        metrics = ['mean_absolute_percentage_error', 'mean_squared_error']

    mlmodel.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # serialize model to JSON
    model_json = mlmodel.to_json()

    if hparams['job_dir'].startswith('gs://'):
        with open(model_json_name, "w") as json_file:
            json_file.write(model_json)
        copy_file_to_gcs(model_folder, model_json_name)
    else:
        with io.open(os.path.join(model_folder, model_json_name), "w") as json_file:
            json_file.write(model_json)

    earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                              min_delta=0.005,
                                                              patience=10,
                                                              verbose=1,
                                                              mode='auto')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                          histogram_freq=0,
                                                          write_graph=True,
                                                          embeddings_freq=0,
                                                          profile_batch=0)

    save_weights_only = False
    if hparams['model_type'] == 'darnn' or hparams['input_attention']:
        save_weights_only = True
    checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                              monitor='val_loss',
                                                              verbose=0,
                                                              save_best_only=True,
                                                              save_weights_only=save_weights_only,
                                                              mode='min',
                                                              save_freq='epoch')


    class CustomMetricsValTest(tf.keras.callbacks.Callback):
        def __init__(self, train_data, val_data, test_data, validation_freq):
            super().__init__()
            self.train_data = train_data
            self.validation_data = val_data
            self.test_data = test_data
            self.validation_freq = validation_freq

        def on_train_begin(self, logs={}):
            self._data = []

        def on_epoch_end(self, epoch, logs={}):

            one_indexed_epoch = epoch + 1

            if one_indexed_epoch % self.validation_freq == 0:

                # Train data
                predict = np.squeeze(np.asarray(self.model.predict(self.train_data[0])), axis=-1)
                targ = tf.cast(self.train_data[1], tf.float32)
                epoch_prediction_of_change_in_direction = prediction_of_change_in_direction(targ, predict)
                epoch_prediction_of_change_in_direction_ret = prediction_of_change_in_direction_ret(targ, predict)
                epoch_u_theil = u_theil(targ, predict)
                epoch_average_relative_variance = average_relative_variance(targ, predict)
                epoch_index_agreement = index_agreement(targ, predict)
                epoch_sum_of_losses_and_gains = sum_of_losses_and_gains(targ, predict)
                epoch_sum_of_losses_and_gains_ret = sum_of_losses_and_gains_ret(targ, predict)

                # Val data
                predict = np.squeeze(np.asarray(self.model.predict(self.validation_data[0])), axis=-1)
                targ = tf.cast(self.validation_data[1], tf.float32)
                epoch_val_prediction_of_change_in_direction = prediction_of_change_in_direction(targ, predict)
                epoch_val_prediction_of_change_in_direction_ret = prediction_of_change_in_direction_ret(targ, predict)
                epoch_val_u_theil = u_theil(targ, predict)
                epoch_val_average_relative_variance = average_relative_variance(targ, predict)
                epoch_val_index_agreement = index_agreement(targ, predict)
                epoch_val_sum_of_losses_and_gains = sum_of_losses_and_gains(targ, predict)
                epoch_val_sum_of_losses_and_gains_ret = sum_of_losses_and_gains_ret(targ, predict)

                # Test data
                predict = np.squeeze(np.asarray(self.model.predict(self.test_data[0])), axis=-1)
                targ = tf.cast(self.test_data[1], tf.float32)
                epoch_test_mse = mse_custom(targ, predict)
                epoch_test_mape = mape_custom(targ, predict)
                epoch_test_prediction_of_change_in_direction = prediction_of_change_in_direction(targ, predict)
                epoch_test_prediction_of_change_in_direction_ret = prediction_of_change_in_direction_ret(targ, predict)
                epoch_test_u_theil = u_theil(targ, predict)
                epoch_test_average_relative_variance = average_relative_variance(targ, predict)
                epoch_test_index_agreement = index_agreement(targ, predict)
                epoch_test_sum_of_losses_and_gains = sum_of_losses_and_gains(targ, predict)
                epoch_test_sum_of_losses_and_gains_ret = sum_of_losses_and_gains_ret(targ, predict)

                # Write summaries for tensorboard
                tf.summary.scalar('epoch_POCID', data=epoch_prediction_of_change_in_direction, step=epoch)
                tf.summary.scalar('epoch_val_POCID', data=epoch_val_prediction_of_change_in_direction, step=epoch)
                tf.summary.scalar('epoch_test_POCID', data=epoch_test_prediction_of_change_in_direction, step=epoch)
                tf.summary.scalar('epoch_POCID_ret', data=epoch_prediction_of_change_in_direction_ret, step=epoch)
                tf.summary.scalar('epoch_val_POCID_ret', data=epoch_val_prediction_of_change_in_direction_ret, step=epoch)
                tf.summary.scalar('epoch_test_POCID_ret', data=epoch_test_prediction_of_change_in_direction_ret, step=epoch)
                tf.summary.scalar('epoch_U_THEIL', data=epoch_u_theil, step=epoch)
                tf.summary.scalar('epoch_val_U_THEIL', data=epoch_val_u_theil, step=epoch)
                tf.summary.scalar('epoch_test_U_THEIL', data=epoch_test_u_theil, step=epoch)
                tf.summary.scalar('epoch_ARV', data=epoch_average_relative_variance, step=epoch)
                tf.summary.scalar('epoch_val_ARV', data=epoch_val_average_relative_variance, step=epoch)
                tf.summary.scalar('epoch_test_ARV', data=epoch_test_average_relative_variance, step=epoch)
                tf.summary.scalar('epoch_IA', data=epoch_index_agreement, step=epoch)
                tf.summary.scalar('epoch_val_IA', data=epoch_val_index_agreement, step=epoch)
                tf.summary.scalar('epoch_test_IA', data=epoch_test_index_agreement, step=epoch)
                tf.summary.scalar('epoch_SLG', data=epoch_sum_of_losses_and_gains, step=epoch)
                tf.summary.scalar('epoch_val_SLG', data=epoch_val_sum_of_losses_and_gains, step=epoch)
                tf.summary.scalar('epoch_test_SLG', data=epoch_test_sum_of_losses_and_gains, step=epoch)
                tf.summary.scalar('epoch_SLG_ret', data=epoch_sum_of_losses_and_gains_ret, step=epoch)
                tf.summary.scalar('epoch_val_SLG_ret', data=epoch_val_sum_of_losses_and_gains_ret, step=epoch)
                tf.summary.scalar('epoch_test_SLG_ret', data=epoch_test_sum_of_losses_and_gains_ret, step=epoch)
                tf.summary.scalar('epoch_test_MSE', data=epoch_test_mse, step=epoch)
                tf.summary.scalar('epoch_test_MAPE', data=epoch_test_mape, step=epoch)

            return

        def get_data(self):
            return self._data

    custom_metrics = CustomMetricsValTest([train_samples, train_targets],
                                          [val_samples, val_targets],
                                          [test_samples, test_targets],
                                          validation_freq=hparams['validation_freq'])

    # log hparams
    list_hparamas = hparams.keys()

    hparams_new = []
    for i in list_hparamas:
        globals()[i] = hp.HParam(i)
        hparams_new.append(globals()[i])

    hparam_callback = hp.KerasCallback(logdir, hparams)

    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(
            hparams=hparams_new,
            metrics=[hp.Metric('epoch_loss', group="validation", display_name='Loss (val.)'),
                     hp.Metric('epoch_loss', group="train", display_name='Loss (train.)'),
                     hp.Metric('epoch_mean_squared_error', group="validation", display_name='MSE (val.)'),
                     hp.Metric('epoch_mean_squared_error', group="train", display_name='MSE (train.)'),
                     hp.Metric('epoch_mean_absolute_percentage_error', group="validation", display_name='MAPE (val.)'),
                     hp.Metric('epoch_mean_absolute_percentage_error', group="train", display_name='MAPE (train.)'),
                     hp.Metric('epoch_POCID', group="metrics", display_name='POCID (train.)'),
                     hp.Metric('epoch_val_POCID', group="metrics", display_name='POCID (val.)'),
                     hp.Metric('epoch_test_POCID', group="metrics", display_name='POCID (test.)'),
                     hp.Metric('epoch_POCID_ret', group="metrics", display_name="POCID_ret (train.)"),
                     hp.Metric('epoch_val_POCID_ret', group="metrics", display_name="POCID_ret (val.)"),
                     hp.Metric('epoch_test_POCID_ret', group="metrics", display_name="POCID_ret (test.)"),
                     hp.Metric('epoch_U_THEIL', group="metrics", display_name="U_THEIL (train.)"),
                     hp.Metric('epoch_val_U_THEIL', group="metrics", display_name="U_THEIL (val.)"),
                     hp.Metric('epoch_test_U_THEIL', group="metrics", display_name="U_THEIL (test.)"),
                     hp.Metric('epoch_ARV', group="metrics", display_name="ARV (train.)"),
                     hp.Metric('epoch_val_ARV', group="metrics", display_name="ARV (val.)"),
                     hp.Metric('epoch_test_ARV', group="metrics", display_name="ARV (test.)"),
                     hp.Metric('epoch_IA', group="metrics", display_name="IA (train.)"),
                     hp.Metric('epoch_val_IA', group="metrics", display_name="IA (val.)"),
                     hp.Metric('epoch_test_IA', group="metrics", display_name="IA (test.)"),
                     hp.Metric('epoch_SLG', group="metrics", display_name="SLG (train.)"),
                     hp.Metric('epoch_val_SLG', group="metrics", display_name="SLG (val.)"),
                     hp.Metric('epoch_test_SLG', group="metrics", display_name="SLG (train.)"),
                     hp.Metric('epoch_SLG_ret', group="metrics", display_name="SLG_ret (val.)"),
                     hp.Metric('epoch_val_SLG_ret', group="metrics", display_name="SLG_ret (test.)"),
                     hp.Metric('epoch_test_SLG_ret', group="metrics", display_name="SLG_ret (train.)"),
                     hp.Metric('epoch_test_MSE', group="metrics", display_name="MSE (test.)"),
                     hp.Metric('epoch_test_MAPE', group="metrics", display_name="MAPE (test.)")
                     ]
        )

    mlmodel.summary()

    mlmodel.fit(x=train_samples, y=train_targets,
                batch_size=hparams['batch_size'],
                steps_per_epoch=hparams['steps_per_epoch'],
                epochs=hparams['num_epochs'],
                validation_data=[val_samples, val_targets],
                verbose=2,
                validation_freq=hparams['validation_freq'],
                callbacks=[#earlystopping_callback,
                           custom_metrics,
                           tensorboard_callback,
                           checkpoints_callback,
                           hparam_callback
                            ])

if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    # tf.compat.v1.disable_eager_execution()

    # hparams = hparam.HParams(**args.__dict__)
    hparams = args.__dict__
    train_and_evaluate(hparams)
