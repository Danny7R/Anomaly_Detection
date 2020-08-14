import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, LSTM
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
K.set_image_data_format('channels_last')
from data import load_data_3

m = 19
x1_h1, _, _, _, y1_h1, _, _, _ = load_data_3(leak_size='*', inx='all', m=19, d=1)
print(x1_h1.shape, y1_h1.shape)


def model1(input_shape):
    model_input = Input(input_shape)
    x = LSTM(20)(model_input)  # , return_sequences=True
    # x = LSTM(10, return_sequences=True)(x)
    # x = LSTM(20)(x)
    # x = Flatten()(x)
    output = Dense(4, activation='linear')(x)
    model = Model(inputs=model_input, outputs=output, name='model1')
    return model


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


m1 = model1((m, 4))
# adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
m1.compile(optimizer='adam', loss='mse', metrics=[rmse])
history = m1.fit(x1_h1, y1_h1, epochs=200, batch_size=32, validation_split=0.15, verbose=2)
m1.save('lstm_all_leaks_13.h5')
