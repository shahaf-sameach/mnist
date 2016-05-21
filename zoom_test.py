import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from scipy.ndimage.interpolation import rotate, shift
from keras.datasets import mnist
from keras.utils import np_utils


# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


def rand_jitter(temp):
    if np.random.random() > .7:
        temp[np.random.randint(0,28,1), :] = 0
    if np.random.random() > .7:
        temp[:, np.random.randint(0,28,1)] = 0
    if np.random.random() > .7:
        temp = shift(temp, shift=(np.random.randint(-3,3,2)))
    if np.random.random() > .7:
        temp = rotate(temp, angle = np.random.randint(-20,20,1), reshape=False)
    return temp


import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.figure()
f, ax = plt.subplots(2, 2)
ax[0,0].imshow(X_test[1,0,:,:])
ax[0,1].imshow(zoom(X_test[1,0,:,:],10))
ax[1,0].imshow(zoom(X_test[1,0,:,:],20))
ax[1,1].imshow(zoom(X_test[1,0,:,:],30))
plt.show()

# Copy to not effect the original
ind = np.random.randint(len(X_train))
test_image = lambda : np.copy(X_train[ind,0,:,:])



