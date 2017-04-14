'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from jitter import rand_jitter
from time import time
from network import Network

def write_to_file(predictions, filename='predictions.txt'):
  with open(filename, 'w') as predictions_file:
    for predict in predictions:
      predictions_file.write("%s\n" % np.argmax(predict))

  print("wrote predictions to %s" % filename)


def load_data_from_file(filename="train.txt"):
  data = np.loadtxt(filename, delimiter=",", converters = {0: lambda s: 0 if s == '?' else s})
  # seperate to X and Y
  x_data = data[:,1:]
  y_data = data[:,0]
  X_data = x_data.reshape(x_data.shape[0], 1, 28, 28)
  X_data = X_data.astype('float32')
  Y_data = np_utils.to_categorical(y_data, 10)

  return X_data, Y_data


if __name__ == '__main__':
  print("Loading data sets...")
  x_data, y_data = load_data_from_file("test.txt")
  print(x_data.shape)
  print("-----------")
  print(y_data.shape)