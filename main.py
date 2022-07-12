import tensorflow 
from tensorflow.keras.models import load_model
from cnn_m_bdlstm import train_CNN_MDLSTM

if __name__ == '__main__':
    loading_model = False
    if loading_model:
        model = load_model('ABC.h5')
    else:
        model = None
    sequence_length = 1440 #120  # past minutes 
    prediction_steps = 60 #60  # future minutes 
    train_CNN_MDLSTM(model , sequence_length, prediction_steps)
