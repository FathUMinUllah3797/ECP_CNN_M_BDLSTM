import time
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import TimeDistributed
from math import sqrt
from dataset import load_data
from numpy.random import seed
seed(1234)  # seed random numbers for Keras

#from tensorflow import set_random_seed
tf.random.set_seed(2)  # seed random numbers for Tensorflow backend
from plot import plot_predictions
from matplotlib import pyplot
import numpy as np


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

#def mse(predicted, observed):
 #   return np.sum(np.multiply((predicted - observed),(predicted - observed)))/predicted.shape[0]


def train_CNN_MDLSTM(model, sequence_length, prediction_steps):
    data = None
    global_start_time = time.time()
    epochs = 20
    batch=100
    ratio_of_data = 1  # ratio of data to use from 2+ million data points
    dataset_path = ''

    if data is None:
        print('Loading data... ')
        
        
        
        x_train, y_train, x_test, y_test, result_mean = load_data(dataset_path, sequence_length,
                                                                  prediction_steps, ratio_of_data)
    else:
        #print ('inside else statement')
        x_train, y_train, x_test, y_test = data

    print('\nData Loaded. Compiling...\n')

    if model is None:
        model = develop_model(prediction_steps)
        kf = KFold(n_splits=10, shuffle=True)
        try:
            for index, (train_indices, val_indices) in enumerate(kf.split(x_train, y_train)):
                print("Training on fold " + str(index + 1) + "/10....")
                xtrain, x_test = x_train[train_indices], x_train[val_indices]
                ytrain, y_test = y_train[train_indices], y_train[val_indices]
                model.fit(xtrain, ytrain, batch_size=batch, epochs=epochs)
                predicted = model.predict(x_test)
                print("Test mse is : ", mean_squared_error(predicted,y_test))
                mse=mean_squared_error(predicted,y_test)
                rmse=sqrt(mse)
                print("Test rmse is : ", rmse)
                # predicted = np.reshape(predicted, (predicted.size,))
                model.save('LSTM_power_consumption'+ str(index)+'_model.h5')  # save LSTM model
                #print(model.summmary())
        except KeyboardInterrupt:  # save model if training interrupted by user
            print('Duration of training (s) : ', time.time() - global_start_time)
            model.save('Models/CNN-MDLSTM_power_consumption_model' + str(index) + '.h5')
            return model, y_test, 0
    else:  # previously trained mode is given
        print('Loading model...')
        print(model.summary())
        predicted = model.predict(x_test)
    plot_predictions(result_mean, prediction_steps, predicted, y_test, global_start_time)
    #pyplot.plot(history.history['mean_squared_error'])
    #pyplot.plot(history.history['mean_absolute_error'])
    #pyplot.plot(history.history['mean_absolute_percentage_error'])
    #pyplot.show()

    return None

