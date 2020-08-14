import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
import tsa
from data import load_data_3
from datetime import datetime
# from packaging import version
import tensorboard
tensorboard.__version__


m = 19
x1_h1, _, _, _, y1_h1, _, _, _ = load_data_3(leak_size='*', inx='all', m=19, d=1)
print(x1_h1.shape, y1_h1.shape)


def model1(input_shape):
    model_input = Input(input_shape)

    x = Conv1D(32, 3, strides=1, padding='same', kernel_initializer='he_normal', activation='relu')(model_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x)
    # x = Dropout(0.2)(x)
    x = Conv1D(64, 3, strides=1, padding='same', kernel_initializer='he_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x)
    # x = Conv1D(128, 3, strides=1, padding='same', kernel_initializer='he_normal', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    output = Dense(4, activation='linear')(x)
    model = Model(inputs=model_input, outputs=output, name='model1')
    return model


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


m1 = model1((m, 4))
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
m1.compile(optimizer='adam', loss="mse", metrics=[rmse])

#  TensorBoard callback
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

history = m1.fit(x1_h1, y1_h1, epochs=2, batch_size=512, validation_split=0.1, verbose=2, callbacks=[tensorboard_callback])
m1.save('cnn_all_leaks_21.h5')
