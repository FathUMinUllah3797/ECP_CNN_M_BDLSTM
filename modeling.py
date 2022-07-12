import time
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import TimeDistributed

def develop_model(prediction_steps):
    model = Sequential()
    layers = [1, 75, 100, prediction_steps]
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(prediction_steps,1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(TimeDistributed(Dense(32,activation='relu')))
    model.add(Bidirectional(LSTM(layers[1], input_shape=(None, layers[0]), return_sequences=True)))  # add first layer
    model.add(Dropout(0.2))  # add dropout for first layer
    model.add(Bidirectional(LSTM(layers[2], return_sequences=False)))  # add second layer
    model.add(Dropout(0.2))  # add dropout for second layer
    model.add(Dense(layers[3]))  # add output layer
    model.add(Activation('linear'))  # output layer with linear activation
    start = time.time()
    model.compile(loss="mean_squared_error", optimizer="rmsprop" , metrics=['mse', 'mae', 'mape'])
    #model.summmary()
    print('Compilation Time : ', time.time() - start)
    return model