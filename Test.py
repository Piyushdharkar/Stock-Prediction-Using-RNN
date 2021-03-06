from keras.models import load_model
import pandas as pd
from DataProcessing import process_data, fitted_scaler
import numpy as np
import matplotlib.pyplot as plt

model = load_model('saved_model.h5')

lookback = 2

dataset = pd.read_csv('dataset.csv')
dataset = dataset.sort_values(by='Date')
dataset = np.array(dataset['Adj Close'])
dataset = np.reshape(dataset, (dataset.shape[0], 1))

train_data = dataset[ : 4001]

scaler = fitted_scaler(train_data)
test_data = dataset[4501 : ]

test_data = scaler.transform(test_data)

x_test, y_test = process_data(test_data, lookback=lookback)
x_test = np.resize(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

y_pred = model.predict(x_test)

y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

print("Actual adjusted closing price")
print(y_test)
print("Predicted adjusted closing price")
print(y_pred)

plt.subplot(331)
plt.plot(y_test, y_pred)
plt.xlabel('Actual adjusted closing price')
plt.scatter(y_test, y_pred)
plt.ylabel('Predicted adjusted closing price')
plt.axis('scaled')





