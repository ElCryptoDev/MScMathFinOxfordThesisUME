""" 
Many-to-many model implementation.

Based on https://github.com/iLampard/dual_stage_attention_rnn/blob/master/da_rnn/model/encoder_decoder.py

"""
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow import keras
from tensorflow.keras.layers import LSTMCell as KerasLSTMCell


class Attention(keras.Model):
    def __init__(self, input_dim, var_scope, reuse=tf.compat.v1.AUTO_REUSE):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.var_scope = var_scope
        with tf.compat.v1.variable_scope(var_scope, reuse=reuse):
            self.attention_w = layers.Dense(self.input_dim, name='W')
            self.attention_u = layers.Dense(self.input_dim, name='U')
            self.attention_v = layers.Dense(1, name='V')

    def call(self, inputs):
        """
        Compute the attention weight for input series
        hidden_state, cell_state (batch_size, hidden_dim)
        input_x (batch_size, num_series, input_dim),
        input_dim = num_steps for input attention
        """
        input_x = inputs[0]
        prev_state_tuple = inputs[1]

        prev_hidden_state, prev_cell_state = prev_state_tuple

        # (batch_size, 1, hidden_dim * 2)
        concat_state = tf.expand_dims(tf.concat([prev_hidden_state, prev_cell_state], axis=-1),
                                      axis=1)

        # (batch_size, num_series, input_dim)
        score_ = self.attention_w(concat_state) + self.attention_u(input_x)

        # (batch_size, num_series, 1)
        # Equation (8)
        score = self.attention_v(tf.nn.tanh(score_))

        # (batch_size, num_series)
        # Equation (9)
        weight = tf.squeeze(tf.nn.softmax(score, axis=1), axis=-1)

        return weight

    def get_config(self, verbose=0):
        '''Return the configuration of the model
        as a dictionary.

        To load a model from its configuration, use
        `keras.models.model_from_config(config, custom_objects={})`.
        '''
        config = {
            'input_dim': self.input_dim,
            'var_scope': self.var_scope
        }
        return config


class Encoder(layers.Layer):
    def __init__(self, encoder_dim, num_steps):
        super(Encoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.num_steps = num_steps
        self.attention_layer = Attention(num_steps, var_scope='input_attention')
        self.lstm_cell = KerasLSTMCell(encoder_dim)

    def call(self, inputs):
        """
        inputs: (batch_size, num_steps, num_series)
        """

        def one_step(prev_state_tuple, current_input):
            """ Move along the time axis by one step  """

            # (batch_size, num_series, num_steps)
            inputs_scan = tf.transpose(inputs, perm=[0, 2, 1])

            # (batch_size, num_series)
            weight = self.attention_layer([inputs_scan, prev_state_tuple])

            weighted_current_input = weight * current_input

            #return self.lstm_cell(weighted_current_input, prev_state_tuple)
            out, [h, c] = self.lstm_cell(weighted_current_input, prev_state_tuple)
            return (out, c)

        # Get the batch size from inputs
        self.batch_size = tf.shape(inputs)[0]
        self.num_steps = inputs.get_shape().as_list()[1]

        self.init_hidden_state = tf.compat.v1.random_normal([self.batch_size, self.encoder_dim])
        self.init_cell_state = tf.compat.v1.random_normal([self.batch_size, self.encoder_dim])

        # (num_steps, batch_size, num_series)
        inputs_ = tf.transpose(inputs, perm=[1, 0, 2])

        # use scan to run over all time steps
        state_tuple = tf.scan(one_step,
                              elems=inputs_,
                              initializer=(self.init_hidden_state,
                                           self.init_cell_state))

        # (batch_size, num_steps, encoder_dim)
        all_hidden_state = tf.transpose(state_tuple[0], perm=[1, 0, 2])
        return all_hidden_state

    def get_config(self, verbose=0):
        '''Return the configuration of the model
        as a dictionary.

        To load a model from its configuration, use
        `keras.models.model_from_config(config, custom_objects={})`.
        '''
        config = {
            'num_steps': self.num_steps,
            'encoder_dim': self.encoder_dim
        }
        return config


class Decoder(layers.Layer):
    def __init__(self, decoder_dim, num_steps, categorical):
        super(Decoder, self).__init__()
        self.decoder_dim = decoder_dim
        self.num_steps = num_steps
        self.attention_layer = Attention(num_steps, var_scope='temporal_attention')
        self.lstm_cell = KerasLSTMCell(decoder_dim)
        self.layer_fc_context = layers.Dense(1)
        self.layer_prediction_fc_1 = layers.Dense(decoder_dim)
        if categorical:
            self.layer_prediction_fc_2 = layers.Dense(1, activation='sigmoid')
        else:
            self.layer_prediction_fc_2 = layers.Dense(1)

    def call(self, inputs):
        """
        encoder_states: (batch_size, num_steps, encoder_dim)
        labels: (batch_size, num_steps)
        """
        encoder_states = inputs[0]
        labels = inputs[1]

        def one_step(accumulator, current_label):
            """ Move along the time axis by one step  """

            prev_state_tuple, context = accumulator
            # (batch_size, num_steps)
            # Equation (12) (13)
            weight = self.attention_layer([encoder_states, prev_state_tuple])

            # Equation (14)
            # (batch_size, encoder_dim)
            context = tf.reduce_sum(tf.expand_dims(weight, axis=-1) * encoder_states,
                                    axis=1)

            # Equation (15)
            # (batch_size, 1)
            y_tilde = self.layer_fc_context(tf.concat([tf.expand_dims(current_label, axis=-1), context], axis=-1))

            # Equation (16)
            #return self.lstm_cell(y_tilde, prev_state_tuple), context
            out, [h, c] = self.lstm_cell(y_tilde, prev_state_tuple)
            return (out, c), context

        # Get the batch size from inputs
        self.batch_size = tf.shape(encoder_states)[0]
        self.num_steps = encoder_states.get_shape().as_list()[1]
        self.encoder_dim = encoder_states.get_shape().as_list()[-1]

        init_hidden_state = tf.compat.v1.random_normal([self.batch_size, self.decoder_dim])
        init_cell_state = tf.compat.v1.random_normal([self.batch_size, self.decoder_dim])
        init_context = tf.compat.v1.random_normal([self.batch_size, self.encoder_dim])

        # (num_steps, batch_size)
        inputs_ = tf.transpose(labels, perm=[1, 0])

        # use scan to run over all time steps
        state_tuple, all_context = tf.scan(one_step,
                                           elems=inputs_,
                                           initializer=((init_hidden_state,
                                                        init_cell_state),
                                                        init_context))

        # (batch_size, num_steps, decoder_dim)
        all_hidden_state = tf.transpose(state_tuple[0], perm=[1, 0, 2])

        # (batch_size, num_steps, encoder_dim)
        all_context = tf.transpose(all_context, perm=[1, 0, 2])

        last_hidden_state = all_hidden_state[:, -1, :]
        last_context = all_context[:, -1, :]

        # (batch_size, 1)
        # Equation (22)
        pred_ = self.layer_prediction_fc_1(tf.concat([last_hidden_state, last_context], axis=-1))
        pred = self.layer_prediction_fc_2(pred_)

        return pred

    def get_config(self, verbose=0):
        '''Return the configuration of the model
        as a dictionary.

        To load a model from its configuration, use
        `keras.models.model_from_config(config, custom_objects={})`.
        '''
        config = {
            'num_steps': self.num_steps,
            'decoder_dim': self.decoder_dim
        }
        return config


def darnn_model(num_values, num_nodes, lookback, input_attention=True, temporal_attention=True, categorical=False):

    input = Input(shape=(lookback, num_values), name='input')
    input_labels = Input(shape=(lookback,), name='input_labels')

    if input_attention==True:
        encoder = Encoder(num_nodes, lookback)
        encoder_states = encoder(input)
    else:
        encoder_states = layers.LSTM(num_nodes, return_sequences=True)(input)

    if temporal_attention==True:
        decoder = Decoder(num_nodes, lookback, categorical)
        pred = decoder([encoder_states, input_labels])
    else:
        lstm_out = layers.LSTM(num_nodes, return_sequences=False)(tf.concat([encoder_states, tf.expand_dims(input_labels, -1)], axis=-1))
        pred = layers.Dense(num_nodes+input_labels.shape[-1])(lstm_out)
        if categorical:
            pred = layers.Dense(1, activation='sigmoid')(pred)
        else:
            pred = layers.Dense(1)(pred)

    model = Model(inputs=[input, input_labels], outputs=[pred])

    return model

