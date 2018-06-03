from sklearn.preprocessing import MinMaxScaler
import numpy as np

def process_data(data, lookback):
    X, Y = [], []
    for i in range(len(data) - lookback - 1):
        X.append(data[i : (i + lookback), 0])
        Y.append(data[(i + lookback), 0])
    return np.array(X), np.array(Y)


def fitted_scaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    
    return scaler