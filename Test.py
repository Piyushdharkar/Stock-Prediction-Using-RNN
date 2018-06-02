from keras.models import load_model
import pandas as pd
from ProcessData import process_data
import numpy as np
from FittedScaler import fitted_scaler

model = load_model('saved_model.h5')

dataset = pd.read_csv('dataset.csv')
dataset = dataset.sort_values(by='Date')
dataset = dataset.drop('Date', axis=1)

test_data = np.array(dataset[4001 : ])

x_test, y_test = process_data(test_data, lookback=10)
x_test = np.resize(x_test, (x_test.shape[0], x_test.shape[1], 1))

train_data = np.array(dataset[ : 3001])

scaler = fitted_scaler(train_data)

