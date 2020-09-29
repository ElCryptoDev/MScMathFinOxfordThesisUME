""" 
Many-to-one model implementation

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, RepeatVector, concatenate, dot, Add, Reshape
from .darnn_model import Encoder


def value_rnn_model(num_values,
                    num_layers,
                    lookback,
                    num_nodes=512,
                    dropout=False,
                    input_attention=False,
                    temporal_attention=False,
                    attention_stlye='luong',
                    categorical=False
                    ):
    global attention_score
    input = Input(shape=(lookback, num_values), name='input')
    prev = input

    # Attentional encoder
    if input_attention==True:
        encoder = Encoder(num_nodes, lookback)
        prev = encoder(input)
        state_h = prev[:, -1, :]
    elif input_attention==False:
    # Classical LSTM layers
        lstm, state_h, state_c  = LSTM(num_nodes, return_sequences=True, return_state=True, name='lstm_layer_1')(prev)
        if dropout:
            prev = Dropout(dropout)(lstm)
        else:
            prev = lstm

    if num_layers >= 2:
        # Classical LSTM layers
        for i in range(1, num_layers):
            lstm, state_h, state_c = LSTM(num_nodes, return_sequences=True, return_state=True, name='lstm_layer_%d' % (i + 1))(prev)
            if dropout:
                prev = Dropout(dropout)(lstm)
            else:
                prev = lstm

    # temporal attention mechanim
    if temporal_attention:
        hidden_size = num_nodes

        if attention_stlye=='luong':
            score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(prev)
            attention_score = dot([score_first_part, state_h], [2, 1], name='attention_score')
        elif attention_stlye=='bahdanau':
            sum_second_part = Dense(hidden_size, use_bias=False, name='sum_first_part')(prev)
            sum_first_part = Dense(hidden_size, use_bias=False, name='sum_second_part')(state_h)
            sum_first_part = RepeatVector(lookback)(sum_first_part)
            score_pre = Add()([sum_first_part, sum_second_part])
            score_tanh = Activation('tanh', name='score_tanh')(score_pre)
            attention_score = Dense(1)(score_tanh)
            attention_score = Reshape((lookback, ), name='attention_score')(attention_score)

        attention_weights = Activation('softmax', name='attention_weight')(attention_score)
        context_vector = dot([prev, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, state_h], name='attention_output')
        prev = Dense(num_nodes)(pre_activation)
    else:
        # Select last hidden state
        prev = prev[:, -1, :]

    if categorical:
        dense = Dense(1, name='dense_output_sigmoid', activation='sigmoid')(prev)
    else:
        dense = Dense(1, name='dense_output')(prev)

    model = Model(inputs=[input], outputs=[dense])

    return model
