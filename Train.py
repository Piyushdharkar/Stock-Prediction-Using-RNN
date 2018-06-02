import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def process_data(data, lookback):
    X, Y = [], []
    for i in range(len(data) - lookback - 1):
        X.append(data[i : (i + lookback), 0])
        Y.append(data[(i + lookback), 0])
    return np.array(X), np.array(Y)

dataset = pd.read_csv('dataset.csv')
dataset = dataset.sort_values(by='Date')

dataset = dataset['Adj Close']
dataset = np.reshape(dataset, (len(dataset), 1))

scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)

train_data = np.array(dataset[ : 3001])
val_data = np.array(dataset[3001 : 4001])  

x_train, y_train = process_data(train_data, lookback=10)
x_val, y_val = process_data(val_data, lookback=10)


input_shape = (x_train.shape[1], 1)


model = Sequential()
model.add(LSTM(50, input_shape=input_shape))
model.add(Dense(1))
model.add(Activation('linear'))

model.compile(optimizer=keras.optimizers.Adam(), loss='mse')











