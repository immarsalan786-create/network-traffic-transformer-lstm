import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout,
                                     MultiHeadAttention, LayerNormalization,
                                     GlobalAveragePooling1D)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    attention_out = MultiHeadAttention(num_heads=4, key_dim=16)(lstm_out, lstm_out)
    attention_out = LayerNormalization()(attention_out + lstm_out)
    pooled = GlobalAveragePooling1D()(attention_out)
    dropped = Dropout(0.3)(pooled)
    outputs = Dense(num_classes, activation='softmax')(dropped)
    return Model(inputs, outputs)
