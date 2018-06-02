from keras.models import load_model, Sequential
import pandas as pd
from ProcessData import process_data
import numpy as np
from FittedScaler import fitted_scaler

model = load_model('saved_model.h5')

dataset = pd.read_csv('dataset.csv')
dataset = dataset.sort_values(by='Date')
dataset = dataset['Adj Close']

train_data = np.array(dataset[ : 3001])

scaler = fitted_scaler(train_data)
test_data = np.array(dataset[4001 : ])

test_data = scaler.transform(test_data)
test_data = np.reshape(test_data, (test_data.shape[0], 1))

x_test, y_test = process_data(test_data, lookback=10)
x_test = np.resize(x_test, (x_test.shape[0], x_test.shape[1], 1))

y_pred = model.predict(x_test)

y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)


print("Actual adjusted closing price")
print(y_test)
print("Predicted adjusted closing price")
print(y_pred)
