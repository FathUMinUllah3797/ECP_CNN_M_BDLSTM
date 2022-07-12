import numpy as np
import csv

def load_data(dataset_path, sequence_length=120, prediction_steps=60, ratio_of_data=1.0):
    max_values = ratio_of_data * 2075259  # 2075259 is the total number of measurements from Dec 2006 to Nov 2010

    # Load data from file
    with open(dataset_path) as file:
        data_file = csv.reader(file, delimiter=";")
        power_consumption = []
        number_of_values = 0
        for line in data_file:
            try:
                power_consumption.append(float(line[2]))
                number_of_values += 1
            except ValueError:
                pass
            if number_of_values >= max_values:  # limit data to be considered by model according to max_values
                break
    print(len(power_consumption))
    #print(power_consumption)
    #print(power_consumption)
    print('Loaded data from csv.')
    windowed_data = []
    # Format data into rolling window sequences
    for index in range(len(power_consumption) - sequence_length):  # for e.g: index=0 => 123, index=1 => 234 etc.
        windowed_data.append(power_consumption[index: index + sequence_length])
    windowed_data = np.array(windowed_data)  # shape (number of samples, sequence length)
    print('window shape is :' ,windowed_data.shape)

    # Center data
    data_mean = windowed_data.mean()
    windowed_data -= data_mean
    print('Center data so mean is zero (subtract each data point by mean of value: ', data_mean, ')')
    print('Data  : ', windowed_data.shape)

    
    train_set_ratio = 0.9
    row = int(round(train_set_ratio * windowed_data.shape[0]))
    print('row=', row)
    #print(row.shape)
    train = windowed_data[:row, :]
    print(train.shape)
    print('train=', train)
    x_train = train[:, :-prediction_steps]  # remove last prediction_steps from train set
    print(x_train.shape)
    print('x_train=', x_train)
    y_train = train[:, -prediction_steps:]  # take last prediction_steps from train set
    print(y_train.shape)
    print('y_train=', y_train)
    x_test = windowed_data[row:, :-prediction_steps]
    print(x_test.shape)
    print('x_test=', x_test)
    y_test = windowed_data[row:, -prediction_steps:]  # take last prediction_steps from test set
    print(y_test.shape)
    print('y_test=', y_test)
   
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print(x_train.shape)
    print(x_train)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    print(x_test.shape)
    print(x_test)
    return [x_train, y_train, x_test, y_test, data_mean]
