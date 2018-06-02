from keras.models import load_model
import pandas as pd
from ProcessData import process_data
import numpy as np

model = load_model('saved_model.h5')

dataset = pd.read_csv('dataset.csv')

test_data = np.array(dataset[4001 : ])

x_test, y_test = process_data(test_data, lookback=10)

