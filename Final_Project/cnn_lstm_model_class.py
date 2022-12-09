from matplotlib import pyplot
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from tensorflow.keras.callbacks import EarlyStopping

class CNN_LSTM():

    def __init__(self,
                n_features=None,
                epochs=250,
                learning_rate=0.005,
                dp=None,
                cnn_layers=1,
                n_steps_in=24,
                n_steps_out=2,
                cnn_units_1=100, 
                cnn_units_2=100,
                sub_seq = 2, 
                lstm_units=200,
                chart_name="test"
                ) -> None:
        assert n_features != None
        self.n_features = n_features,
        self.epochs=epochs, 
        self.learning_rate=learning_rate,
        self.dp=dp,
        self.cnn_layers=cnn_layers,
        self.n_steps_in=n_steps_in,
        self.n_steps_out=n_steps_out,
        self.cnn_units_1=cnn_units_1, 
        self.cnn_units_2=cnn_units_2,
        self.sub_seq =sub_seq, 
        self.lstm_units=lstm_units,
        self.chart_name=chart_name
        self.model = None
        self.n_steps = int(self.n_steps_in[0]/self.sub_seq[0])

    
    def load_model(self):
        model = Sequential()
        model.add(TimeDistributed(Conv1D(self.cnn_units_1[0], 1, activation='relu'), input_shape=(None, self.n_steps, self.n_features[0])))
        model.add(TimeDistributed(MaxPooling1D()))
        if self.cnn_layers == 2:
            model.add(TimeDistributed(Conv1D(self.cnn_units_2[0], 1, activation='relu')))
            model.add(TimeDistributed(MaxPooling1D()))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(self.lstm_units[0], activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(self.n_steps_out[0], activation='relu'))
        self.model = model
        return model

    def compile_model(self):
        model = self.model
        compile_model = model.compile(loss='mean_squared_error',
                                    optimizer=tf.keras.optimizers.Adam(self.learning_rate[0]), 
                                    metrics=[tf.keras.metrics.RootMeanSquaredError()])
        return compile_model

    def train_model(self, X_train, y_train, X_dev, y_dev, batch_size=64,verbose=2):
        model = self.model
        # Add early stoping to model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001, mode="auto")
        history = model.fit(X_train, 
                                y_train, 
                                epochs=self.epochs[0], 
                                batch_size=batch_size,
                                validation_data=(X_dev, y_dev),
                                verbose=verbose,
                                callbacks=[early_stopping])
        
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.title("loss and val_loss vs. epochs")
        pyplot.xlabel("epochs")
        pyplot.ylabel("loss")
        pyplot.savefig("experiments/"+self.chart_name+".png")
        pyplot.show()
        return history

    def evaluate_model(self, X, y, batch_size, verbose):
        model = self.model
        evaluate_model = model.evaluate(X,
                                    y,
                                    batch_size=batch_size,
                                    verbose=verbose)
        return evaluate_model

    def predict_y (self, X_dev, verbose):
        model = self.model
        y_hat = model.predict(X_dev, verbose=verbose)
        return y_hat