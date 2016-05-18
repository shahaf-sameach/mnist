from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
# from keras.utils.visualize_util import plot

batch_size = 128
nb_classes = 10
nb_epoch = 50

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3


m = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(m)

std = np.std(m, axis=1)
mean = np.mean(m, axis=1)

a = m[:]

print(a)

