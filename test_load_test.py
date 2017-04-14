import numpy as np
from keras.utils import np_utils

data = np.loadtxt("test.txt", delimiter=",", converters = {0: lambda s: 0})
x_data = data[:,1:]
y_data = data[:,0]
print x_data.shape
X_data = x_data.reshape(x_data.shape[0], 1, 28, 28)
X_data = X_data.astype('float32')
print y_data
Y_train = np_utils.to_categorical(y_data, 10)
print Y_train
