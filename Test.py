from keras.models import load_model
import pandas as pd

model = load_model('saved_model.h5')

dataset = pd.read_csv('dataset.csv')

test_dataset = dataset[4001 : ]
