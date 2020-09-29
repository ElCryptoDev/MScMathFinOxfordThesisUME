""" Self-attention model implementation

The code is based on https://www.tensorflow.org/tutorials/text/transformer

"""

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
import numpy as np

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def split_heads(x, num_heads, depth, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, num_heads, depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])


def scaled_dot_product_attention(q, k, v):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output

def self_attn_model(num_values, d_model, num_heads, lookback, batch_size, full_enc_pred=True, categorical=False):

    depth = d_model // num_heads

    input = Input(shape=(lookback, num_values), batch_size=batch_size, name='input')

    # Create Encoding via dense layer
    input_enc = Dense(d_model, use_bias=False)(input)

    pos_encoding = positional_encoding(10000, d_model)
    # Add positional encoding to the input embedding
    input_enc += pos_encoding[:, :lookback, :]

    # Create encodings for query, key, and values
    query_enc = Dense(d_model, use_bias=False)(input_enc)
    key_enc = Dense(d_model, use_bias=False)(input_enc)
    value_enc = Dense(d_model, use_bias=False)(input_enc)

    # Multi-head attention
    q = split_heads(query_enc, num_heads, depth, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = split_heads(key_enc, num_heads, depth, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = split_heads(value_enc, num_heads, depth, batch_size) # (batch_size, num_heads, seq_len_v, depth)

    scaled_attention = scaled_dot_product_attention(q, k, v)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, d_model))  # (batch_size, seq_len_q, d_model)

    mutli_head_output = Dense(d_model, use_bias=False)(concat_attention)  # (batch_size, seq_len_q, d_model)

    if full_enc_pred:
        # Concat self_embeddings and apply dense relu layer
        concat_output = tf.reshape(mutli_head_output, (batch_size, -1))
        dense_output = Dense(15, activation='relu')(concat_output)
    else:
        # Only use last self-embedding
        dense_output = Dense(15, activation='relu')(mutli_head_output[:, -1, :])

    if categorical:
        pred = Dense(1, name='dense_output_sigmoid', activation='sigmoid')(dense_output)
    else:
        pred = Dense(1, name='dense_output')(dense_output)

    model = Model(inputs=[input], outputs=[pred])

    return model
