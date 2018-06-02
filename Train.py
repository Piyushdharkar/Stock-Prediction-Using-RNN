import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
import numpy as np
import keras
from ProcessData import process_data
from FittedScaler import fitted_scaler

dataset = pd.read_csv('dataset.csv')
dataset = dataset.sort_values(by='Date')

dataset = dataset['Adj Close']
dataset = np.reshape(dataset, (len(dataset), 1))

scaler = fitted_scaler(dataset)
dataset = scaler.transform(dataset)

train_data = np.array(dataset[ : 4001])
val_data = np.array(dataset[4001 : 4501])  

x_train, y_train = process_data(train_data, lookback=10)
x_val, y_val = process_data(val_data, lookback=10)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

input_shape = (x_train.shape[1], 1)
batch_size = 100
epochs = 20

model = Sequential()
model.add(LSTM(60, input_shape=input_shape))
model.add(Dropout(0.8))
model.add(Dense(1))
model.add(Activation('linear'))

model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
         validation_data=(x_val, y_val))

model.save('saved_model.h5')







