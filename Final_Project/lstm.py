import os

from utils import train_dev_test_split
from lstm_model_class import LSTM 

import argparse

"""
Instrictions: Run scrpit to run default model or use the command line to specify parameters.
Example run:
    python3 lstm.py -lstm_units 100 100 -drop_out 0.5 0.5 -learning_rate=0.01 -lstm_layers=2 -chart_name="test_1"
"""

# TODO: Switch this to the directory with the csv file
directory_processed = '/Users/josuesolanoromero/Downloads/Undergrade_Work/Senior_Work/CS_230/Final_Project/ffill_bfill'

# Flags used to run experiments
parser = argparse.ArgumentParser()
parser.add_argument("-epochs","--epochs", help="Epochs for trining", type=int, default=250)
parser.add_argument("-learning_rate","--learning_rate", help="Learning for trining", type=float, default=0.005)
parser.add_argument("-dp","--dir", help="Directory to dataset", type=str, default=directory_processed)
parser.add_argument("-lstm_layers", "--lstm_layers", help="Number LSTM layers (Max 3)", type=int, default=1)
parser.add_argument("-lstm_units", "--lstm_units", nargs="+", help="Number of units in LSTM layers",default=[100, 100, 100])
parser.add_argument("-drop_out", "--drop_out", nargs="+", help="Drop out per LSTM layer", default=[0.5, 0.5, 0.5])
parser.add_argument("-n_steps_in", "--n_steps_in", help="Number of in steps", type=int, default=24)
parser.add_argument("-n_steps_out", "--n_steps_out", help="Number of steps predicted", type=int, default=2)
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

    # create model 
    cnn_lstm = LSTM(
                n_features=n_features,
                epochs=ARGS.epochs,
                learning_rate=ARGS.learning_rate,
                lstm_layers=ARGS.lstm_layers,
                lstm_units=ARGS.lstm_units,
                drop_out=ARGS.drop_out,
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
    assert ARGS.lstm_layers <= 3
    run_experiment(ARGS.dir)

if __name__ == "__main__":
    main()
