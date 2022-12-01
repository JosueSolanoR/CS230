import pandas as pd
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from utils import train_dev_test_split

from numpy import array
from numpy import hstack
from collections import defaultdict

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

import argparse
directory_processed = '/Users/josuesolanoromero/Downloads/Undergrade_Work/Senior_Work/CS_230/CNN-LSTM/ffill_bfill'
parser = argparse.ArgumentParser()
parser.add_argument("-epochs","--epochs", help="Epochs for trining", type=int, default=10)
parser.add_argument("-dp","--dir", help="Directory to dataset", type=str, default=directory_processed)
parser.add_argument("-cnn_units", "--cnn_units", help="Number of units in CNN layer", type=int, default=64)
parser.add_argument("-lstm_units", "--lstm_units", help="Number of units in LSTM layer", type=int, default=20)
parser.add_argument("-chart_name", "--chart_name", help="Name of output chart", type=str, default="test")
ARGS = parser.parse_args()

def load_data(directory_processed, n_steps_in, n_steps_out):

    file_paths = []

    for filename in os.scandir(directory_processed):
        file_paths.append(filename.path)

    X_train, X_dev, X_test, y_train, y_dev, y_test = train_dev_test_split(file_paths, 
                                                                            n_steps_in, 
                                                                            n_steps_out, 
                                                                            train_percent=0.9, 
                                                                            dev_percent=0.05)
    experiment_data = {'X_train': X_train,
                        'X_dev': X_dev,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_dev': y_dev,
                        'y_test': y_test}

    return experiment_data


def CNN_LSTM_model(n_steps_in, n_features, n_steps_out):

    model = Sequential()
    model.add(TimeDistributed(Conv1D(ARGS.cnn_units, 1, activation='relu'), input_shape=(None, n_steps_in, n_features)))
    model.add(TimeDistributed(MaxPooling1D()))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(ARGS.lstm_units, activation='relu'))
    model.add(Dense(n_steps_out, activation='relu'))

    return model

def compile_model(model, learning_rate):
    compie_model = model.compile(loss='mean_squared_error',
                                optimizer=tf.keras.optimizers.Adam(learning_rate), 
                                metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return compile_model 

def train_model(model, X_train, y_train, X_dev, y_dev, epochs, batch_size, verbose):

    history = model.fit(X_train, 
                            y_train, 
                            epochs=epochs, 
                            batch_size=batch_size,
                            validation_data=(X_dev, y_dev),
                            verbose=verbose)
    
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.savefig("experiments/"+ARGS.chart_name+".png")
    pyplot.show()
   
    
    return history 

def predict_y (model, X_dev, verbose):
  
    y_hat = model.predict(X_dev, verbose=verbose)
    return y_hat

def evaluate_model(model, X, y, batch_size, verbose):

    evaluate_model = model.evaluate(X,
                                  y,
                                  batch_size=batch_size,
                                  verbose=verbose)
  
    return evaluate_model


def run_experiment(directory_processed, n_steps_in, n_steps_out, n_sub_seq, n_steps, learning_rate, epochs, batch_size, verbose):

    # load data from directory of processed data
    experiment_data = load_data(directory_processed, n_steps_in, n_steps_out)

    # fetch train, dev, and test data
    X_train = experiment_data['X_train']
    X_dev = experiment_data['X_dev']
    X_test = experiment_data['X_test']
    y_train = experiment_data['y_train']
    y_dev = experiment_data['y_dev']
    y_test = experiment_data['y_test'] 

    # Extracting number of raw features 
    n_features = X_train.shape[2]

    # reshape X from [n_samples, n_steps_in, n_features] into [n_samples, n_sub_seq, n_steps, n_features]
    n_samples = X_train.shape[0]
    X_train = X_train.reshape((n_samples, n_sub_seq, n_steps, n_features))

    n_samples = X_dev.shape[0]
    X_dev = X_dev.reshape((n_samples, n_sub_seq, n_steps, n_features))

    n_samples = X_test.shape[0]
    X_test = X_test.reshape((n_samples, n_sub_seq, n_steps, n_features))

    # # load model 
    # model = LSTM_model(n_steps_in, n_features, n_steps_out) 
    model = CNN_LSTM_model(n_steps, n_features, n_steps_out)
    # compile model 
    compile_model(model, learning_rate) 
    
    # train model 
    train_model(model, X_train, y_train, X_dev, y_dev,epochs, batch_size, verbose)

    # predict_y_hat 
    y_hat = predict_y(model, X_dev, verbose) 
    
    # evaluate model 
    evaluate_model(model, X_dev, y_dev, batch_size, verbose)

def main():
 
    n_steps_in = 8
    n_steps_out = 4
    n_sub_seq = 2 # Must be divisible by n_steps_in
    assert n_steps_in%n_sub_seq == 0
    n_steps = int(n_steps_in/n_sub_seq)
    learning_rate = 0.005
    epochs = ARGS.epochs
    batch_size = 64 
    verbose = 2
    run_experiment(ARGS.dir, n_steps_in, n_steps_out, n_sub_seq, n_steps, learning_rate, epochs, batch_size, verbose)

if __name__ == "__main__":
    main()
