from sklearn.preprocessing import MinMaxScaler

def fitted_scaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    
    return scaler