import os

from utils import train_dev_test_split
from cnn_lstm_model_class import CNN_LSTM 

import argparse

"""
Instrictions: Run scrpit to run default model or use the command line to specify parameters.
Example run:
    python3 cnn_lstm.py -epochs=70 -learning_rate=0.01 -cnn_units_1=100 -cnn_layers=1 -lstm_units=100 -chart_name="test_1"
"""

# TODO: Switch this to the directory with the csv file
directory_processed = '/Users/josuesolanoromero/Downloads/Undergrade_Work/Senior_Work/CS_230/Final_Project/ffill_bfill'

# Flags used to run experiments
parser = argparse.ArgumentParser()
parser.add_argument("-epochs","--epochs", help="Epochs for trining", type=int, default=250)
parser.add_argument("-learning_rate","--learning_rate", help="Learning for trining", type=float, default=0.005)
parser.add_argument("-dp","--dir", help="Directory to dataset", type=str, default=directory_processed)
parser.add_argument("-cnn_units_1", "--cnn_units_1", help="Number of units in CNN layer", type=int, default=100)
parser.add_argument("-cnn_units_2", "--cnn_units_2", help="Number of units in CNN layer", type=int, default=100)
parser.add_argument("-cnn_layers", "--cnn_layers", help="Number CNN layers (Max 2)", type=int, default=1)
parser.add_argument("-n_sub_seq", "--n_sub_seq", help="Number of subseque", type=int, default=2)
parser.add_argument("-n_steps_in", "--n_steps_in", help="Number of in steps", type=int, default=24)
parser.add_argument("-n_steps_out", "--n_steps_out", help="Number of steps predicted", type=int, default=2)
parser.add_argument("-lstm_units", "--lstm_units", help="Number of units in LSTM layer", type=int, default=200)
parser.add_argument("-chart_name", "--chart_name", help="Name of output chart", type=str, default="test")
parser.add_argument("-batch_size", "--batch_size", help="Batch size for to use in training", type=int, default=64)
parser.add_argument("-verbose", "--verbose", help="Verbose value to use in training", type=int, default=2)
ARGS = parser.parse_args()


def load_data(directory_processed, n_steps_in, n_steps_out):
    """Loads data and splits into training, dev, and test datasets"""
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

def run_experiment(directory_processed):
    """
    Args:
        directory_processed - string contining path to csv file used for training

    returns:
        None - Plots learning loss over epochs and saves chart in experiments/ directory
    """

    # load data from directory of processed data
    experiment_data = load_data(directory_processed, ARGS.n_steps_in, ARGS.n_steps_out)

    # fetch train, dev, and test data
    X_train = experiment_data['X_train']
    X_dev = experiment_data['X_dev']
    X_test = experiment_data['X_test']
    y_train = experiment_data['y_train']
    y_dev = experiment_data['y_dev']
    y_test = experiment_data['y_test'] 

    # Extracting number of raw features 
    n_features = X_train.shape[2]
    n_steps = int(ARGS.n_steps_in/ARGS.n_sub_seq)

    # reshape X from [n_samples, n_steps_in, n_features] into [n_samples, n_sub_seq, n_steps, n_features]
    n_samples = X_train.shape[0]
    X_train = X_train.reshape((n_samples, ARGS.n_sub_seq, n_steps, n_features))

    n_samples = X_dev.shape[0]
    X_dev = X_dev.reshape((n_samples, ARGS.n_sub_seq, n_steps, n_features))

    n_samples = X_test.shape[0]
    X_test = X_test.reshape((n_samples, ARGS.n_sub_seq, n_steps, n_features))

    # create model 
    cnn_lstm = CNN_LSTM(
                n_features=n_features,
                epochs=ARGS.epochs,
                learning_rate=ARGS.learning_rate,
                cnn_layers=ARGS.cnn_layers, 
                n_steps_in=ARGS.n_steps_in,
                n_steps_out=ARGS.n_steps_out,
                cnn_units_1=ARGS.cnn_units_1, 
                cnn_units_2=ARGS.cnn_units_2,
                sub_seq = ARGS.n_sub_seq, 
                lstm_units=ARGS.lstm_units,
                chart_name=ARGS.chart_name)#CNN_LSTM_model(n_steps, n_features, n_steps_out)
    # load model
    cnn_lstm.load_model()
    # compile model 
    cnn_lstm.compile_model() 
    # train model 
    cnn_lstm.train_model(X_train, y_train, X_dev, y_dev, ARGS.batch_size, ARGS.verbose)
    # predict_y_hat 
    y_hat = cnn_lstm.predict_y(X_dev, ARGS.verbose) 
    # evaluate model 
    cnn_lstm.evaluate_model(X_dev, y_dev, ARGS.batch_size, ARGS.verbose)

def main():
    # Must be divisible by n_steps_in
    assert ARGS.n_steps_in%ARGS.n_sub_seq == 0
    assert ARGS.cnn_layers <= 2
    run_experiment(ARGS.dir)

if __name__ == "__main__":
    main()
