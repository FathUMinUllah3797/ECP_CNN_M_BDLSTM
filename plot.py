import time
import numpy as np
import matplotlib.pyplot as plt


# plot results


def plot_predictions(result_mean, prediction_steps, predicted, y_test, global_start_time):
    try:
        test_hours_to_plot = 1
        t0 = 5 # time to start plot of predictions
        skip = 5  # skip prediction plots by specified minutes
        print('Plotting predictions...')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(y_test[:test_hours_to_plot * 60, 0] + result_mean, label='Actual data series')  # plot actual test series

        # plot predicted values from t0 to t0+prediction_steps
        '''
        plt.plot(np.arange(t0 - 1, t0 + prediction_steps),
                 np.insert(predicted[t0, :], 0, y_test[t0 - 1, 0]) + result_mean,
                 color='red', label='t+{0} evolution of predictions'.format(prediction_steps))
        for i in range(t0, test_hours_to_plot * 60, skip):
            t0 += skip
            if t0 + prediction_steps > test_hours_to_plot * 60:  # check plot does not exceed boundary
                break
            plt.plot(np.arange(t0 - 1, t0 + prediction_steps),
                     np.insert(predicted[t0, :], 0, y_test[t0 - 1, 0]) + result_mean, color='red')
'''
        # plot predicted value of t+prediction_steps as series
        plt.plot(predicted[:test_hours_to_plot * 60, prediction_steps - 1] + result_mean,
                 #label='t+{0} prediction series'.format(prediction_steps))
                  label='Prediction series'.format(prediction_steps))

        plt.legend(loc='upper right')
        plt.ylabel('Actual Power (kilowatt)')
        plt.xlabel('Time (minutes)')
        #plt.title('Predictions of {0} minutes'.format(test_hours_to_plot * 60))
        plt.title('Energy consumption predictions of Next hour  ')
        plt.show()
    except Exception as e:
        print(str(e))

    return None