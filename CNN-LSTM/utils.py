import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from numpy import hstack

def split_sequences(sequences, n_steps_in, n_steps_out):
    """
    Args:
        sequences - numpy array of processed dataset(raw inputs and output in the last column)
        n_steps_in - number of time steps to be taken for input to the model
        n_steps_out - number of time steps to be taken for output of the model

    Output:
        X - input features for the model
        y - labeled data of the corresponding input features 
    """
    X, y = list(), list()
    
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        
        if out_end_ix > len(sequences):
            break
                
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
            
    return np.array(X), np.array(y)

def supervised_form_data_one_station(file_path, n_steps_in, n_steps_out):
    """
    Args:
        file_path - file path to station dataset
        n_steps_in - number of time steps to be taken for input to the model
        n_steps_out - number of time steps to be taken for output of the model

    Output:
        X - input features from one station
        y - labeled data from input features 
    """
    df = pd.read_csv(file_path, header=0)
    dataset = np.array(df)
    columns = df.columns
    out = np.array(df[columns[1]])
    out = out.reshape((len(out), 1))
    dataset = hstack((dataset, out))
    X, y = split_sequences(dataset, n_steps_in, n_steps_out)
    return np.array(X), np.array(y)

def normalize(X):
    """
    Args:
        X - Final input features

    Output:
        X - Normalized input features
    """
    n, m, k = X.shape
    num_feature_col = k  # 13 feature columns

    # Normalize X using min-max
    for i in range(num_feature_col):
        col = X[:, :, i].flatten()
        col = (col - np.min(col)) / (np.max(col) - np.min(col))
        col = np.reshape(col, (n, m,))
        X[:, :, i] = col

    return np.array(X) 

def supervised_form_data_N_stations(file_paths, n_steps_in, n_steps_out):
    """
    Args:
        file_paths - List of file paths to datasets
        n_steps_in - number of time steps to be taken for input to the model
        n_steps_out - number of time steps to be taken for output of the model

    Output:
        X - Normalized input features for the model
        y - Normalized input features for the model
    """
    Xn = []  # A set of Xi from each dataset
    yn = []  # A set of yi from each dataset
    for file_path in file_paths:
        Xi, yi = supervised_form_data_one_station(file_path, n_steps_in, n_steps_out)
        Xn.append(Xi)
        yn.append(yi)

    # Concatenate all Xi and yi into one X and y
    X = np.concatenate(Xn, axis=0)
    y = np.concatenate(yn, axis=0)

    # We apply min-max normalization
    X = normalize(X)

    return np.array(X), np.array(y)

def train_dev_test_split(file_paths, n_steps_in, n_steps_out, train_percent=0.9, dev_percent=0.05):
    """
    Args:
        file_paths - List of file paths to datasets
        n_steps_in - number of time steps to be taken for input to the model
        n_steps_out - number of time steps to be taken for output of the model
        train_percent - percantage of data to use for training
        dev_percent - percentage of data to use for development

    Output:
        X_train - inputs used for training
        X_dev - inputs used for development
        X_test - inputs used for testing
        y_train - labels used in training
        y_dev - labels used in development
        y_test - labels used for testing
    """
    filepaths = file_paths
    X, y = supervised_form_data_N_stations(file_paths, n_steps_in, n_steps_out)

    # Shuffle data
    assert len(X) == len(y)
    rand_idx = np.random.permutation(len(X))
    X_shuff = X[rand_idx]
    y_shuff = y[rand_idx]

    # Set X splits
    n,m,k = X.shape
    X_train_split = int(n*train_percent)
    X_dev_split = int(n*dev_percent)

    X_train = X_shuff[:X_train_split]
    X_dev = X_shuff[X_train_split:X_train_split + X_dev_split]  
    X_test = X_shuff[X_train_split + X_dev_split:]

    # Set y splits
    n,m = y.shape

    y_train_split = int(n*train_percent)
    y_dev_split = int(n*dev_percent)

    y_train = y_shuff[:y_train_split]
    y_dev = y_shuff[y_train_split:y_train_split + y_dev_split]  
    y_test = y_shuff[y_train_split + y_dev_split:]

    return X_train, X_dev, X_test, y_train, y_dev, y_test