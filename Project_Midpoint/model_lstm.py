# -*- coding: utf-8 -*-
"""Model_LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tEEklY6VNpNnbm6w5avCwaWwLg3pDB4K
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time

from numpy import array
from numpy import hstack
from collections import defaultdict

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.layers.rnn import Bidirectional

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Pre-processing for datasets

directory = '/content/drive/MyDrive/Colab Notebooks/CS230_Project/csv'
out = '/content/drive/MyDrive/Colab Notebooks/CS230_Project/Data_processed'

# Max percent of missing data per column
DATA_PERCENTAGE = 0.13

kept_sets = []
colums_more_than_percent = defaultdict(lambda: 0)
total_num_sets = 0


for filename in os.scandir(directory):
  total_num_sets += 1

  if filename.is_file():
    df = pd.read_csv(filename.path, header=1)

    columns = df.columns
    # Station Name,From,FT,To,TT,"PM10 (ug/m3)","PM2.5(ug/m3)","AT()","BP(mmHg)","SR(W/mt2)","RH(%)","WD(degree)","RF(mm)","NO(ug/m3)","NOx(ppb)","NO2(ug/m3)","NH3(ug/m3)","SO2(ug/m3)","CO(mg/m3)","Ozone(ug/m3)","Benzene()","Toluene()","Xylene()","MP-Xylene()","Eth-Xylene()"
    drop_columns = [columns[0], columns[1], columns[2], columns[3], columns[4], columns[12], columns[16], columns[20], columns[21], columns[22], columns[23], columns[24]]
    # drop_columns = ["Station Name", "From", "FT", "To", "TT", "RF(mm)", "NH3(ug/m3)", "Benzene()","Toluene()","Xylene()","MP-Xylene()","Eth-Xylene()"]

    df = df.drop(drop_columns, axis=1)  # Each file has one entry at the begineing before column names
    n, m = df.shape 

    keepDataSet = True
    i = 0
    # Count number of nan values in each column
    for (columnName, columnData) in df.iteritems():
      count_nan = columnData.isnull().sum()
      percent_nan = count_nan/n

      if percent_nan > DATA_PERCENTAGE:
        #if columnName != "AT()" and columnName != "BP(mmHg)" and percent_nan > DATA_PERCENTAGE:
        # Keep track of colums with more than 15% nans
        colums_more_than_percent[columnName] += 1
        keepDataSet = False

    if keepDataSet:  # Include set
      # Convert object type to int type
      # using dictionary to convert specific columns
      convert_dict = {"PM10 (ug/m3)":float,"PM2.5(ug/m3)":float,
                      "AT()":float,"BP(mmHg)":float,"SR(W/mt2)":float,
                      "RH(%)":float,"WD(degree)":float,
                      "NO(ug/m3)":float,"NOx(ppb)":float,"NO2(ug/m3)":float,
                      "NH3(ug/m3)":float,"SO2(ug/m3)":float,"CO(mg/m3)":float,
                      "Ozone(ug/m3)":float}

      # Only apply forward fill
      df_ffill = df.copy()
      df_ffill.ffill(axis=0, inplace=True)
      # Fill any missed Nan values with mean of columns
      col_means = df_ffill.mean(axis=0) 
      df_ffill = df_ffill.fillna(col_means)
      os.makedirs(out+"/ffill", exist_ok=True)
      df_ffill.to_csv(out+"/ffill/"+filename.name, index=False)


      # Only apply backward fill
      df_bfill = df.copy()
      df_bfill.bfill(axis="rows", inplace=True)

      # Fill any missed Nan values with mean of columns
      col_means = df_bfill.mean(axis=0)
      df_bfill = df_bfill.fillna(col_means)
      os.makedirs(out+"/bfill", exist_ok=True)
      df_bfill.to_csv(out+"/bfill/"+filename.name, index=False)

      # Apply forward fill and backwardfill
      df_ffill_bfill = df.copy()
      df_ffill_bfill.ffill(axis=0, inplace=True)
      df_ffill_bfill.bfill(axis=0, inplace=True)
      col_means = df_ffill_bfill.mean(axis=0)
      df_ffill_bfill = df_ffill_bfill.fillna(col_means)
      os.makedirs(out+"/ffill_bfill", exist_ok=True)
      df_ffill_bfill.to_csv(out+"/ffill_bfill/"+filename.name, index=False)

      kept_sets.append(filename.name)

print("We keep " + str(len(kept_sets)) + " out of " + str(total_num_sets) + " sets.")
print("These sets included: ", sorted(kept_sets))

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

# LSTM Model evaluation
def LSTM_model(X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size, verbose=0):

  # n_steps_in = number of time steps used in the input
  # n_feature = number of feature used in the input
  # n_steps_out = number of time steps used in the output
  n_steps_in, n_feature, n_steps_out = X_train.shape[1], X_train.shape[2], y_train.shape[1]

  # sequential model creation
  model = Sequential()
  model.add(Bidirectional(LSTM(200, input_shape=(n_steps_in, n_feature))))
  model.add(Dropout(0.5))
  #model.add(Bidirectional(LSTM(20)))
  #model.add(Dropout(0.5))
  #model.add(Bidirectional(LSTM(15)))
  #model.add(Dropout(0.3))
  model.add(Dense(2, activation='relu'))

  
  # compile model
  # metrics=[tf.keras.metrics.RootMeanSquaredError()]
  model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=[tf.keras.metrics.RootMeanSquaredError()])

  # fit the model 
  history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

  #plt.plot(history.history[tf.keras.metrics.RootMeanSquaredError()])

  # predict the model
  y_hat = model.predict(X_val, verbose=0)

  # evaluate the model
  rmse = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=verbose)

  return rmse, y_hat

# Train model
directory_processed = '/content/drive/MyDrive/Colab Notebooks/CS230_Project/Data_processed/ffill_bfill'

file_paths = []

for filename in os.scandir(directory_processed):
  file_paths.append(filename.path)

# choose number of stations for training
n = 19
# file_path_processed contains data of n stations  
file_path_processed = [file_paths[i] for i in range(n)]

# number of input time steps
n_steps_in = 5

# number of output time steps
n_steps_out = 2 


print("Number of Stations used for training ", len(file_path_processed))
print('.......')
print()
print(file_path_processed)

# load the training, development, and testing set
X_train, X_dev, X_test, y_train, y_dev, y_test = train_dev_test_split(file_path_processed, n_steps_in, n_steps_out, train_percent=0.9, dev_percent=0.05)

print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of X_dev: ", X_dev.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_test: ", y_test.shape)
print("Shape of y_dev: ", y_dev.shape)

# model hyperparameters
epochs = 250
batch_size = 64
learning_rate = 5e-3 
verbose = 0

# Run model on X_train 
start = time.time()
rmse_train, y_hat_train = LSTM_model(X_train, y_train, X_train, y_train, epochs, learning_rate, batch_size, verbose=0)
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

print()
print('Evaluation on Training Set: ')
print('RMSE_train:', rmse_train)
print("predicted y_hat_train : ", y_hat_train[25])
print("y_hat_train shape: ", y_hat_train.shape)
print("Original y_train: ", y_train[25])

# Run model on X_dev
start = time.time()
rmse_dev, y_hat_dev = LSTM_model(X_train, y_train, X_dev, y_dev, epochs, learning_rate, batch_size, verbose=0)
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

print('Evaluation on Development Set: ')
print()
print('RMSE_dev:', rmse_dev)
print("predicted y_hat_dev : ", y_hat_dev[25])
print("y_hat_dev shape: ", y_hat_dev.shape)
print("Original y_dev: ", y_dev[25])
