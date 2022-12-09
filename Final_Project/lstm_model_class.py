from matplotlib import pyplot
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM as l
from keras.layers import Dropout

from tensorflow.keras.callbacks import EarlyStopping

class LSTM():

    def __init__(self,
                n_features=None,
                epochs=250,
                learning_rate=0.005,
                dp=None,
                lstm_layers=1,
                n_steps_in=24,
                n_steps_out=2,
                lstm_units=[100,100,100],
                drop_out=[0.5,0.5,0.5],
                chart_name="test"
                ) -> None:
        assert n_features != None
        self.n_features = n_features,
        self.epochs=epochs, 
        self.learning_rate=learning_rate,
        self.dp=dp,
        self.lstm_layers=lstm_layers,
        self.n_steps_in=n_steps_in,
        self.n_steps_out=n_steps_out,
        self.lstm_units=lstm_units,
        self.drop_out=drop_out,
        self.chart_name=chart_name
        self.model = None
    
    def load_model(self):
        model = Sequential()
        model.add(l(int(self.lstm_units[0][0]), input_shape=(self.n_steps_in[0], self.n_features[0])))
        model.add(Dropout(float(self.drop_out[0][0])))
        if self.lstm_layers == 2:
            model.add(l(int(self.lstm_units[0][1]), return_sequences=True))
            model.add(Dropout(float(self.drop_out[0][1])))
        elif self.lstm_layers == 3:
            model.add(l(int(self.lstm_units[0][2])))
            model.add(Dropout(float(self.drop_out[0][2])))        
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