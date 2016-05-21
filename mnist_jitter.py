'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import sys
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from scipy.ndimage.interpolation import rotate, shift, zoom
from sklearn.cross_validation import train_test_split
import time
# from keras.utils.visualize_util import plot

batch_size = 32
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

train_data = np.loadtxt("train.txt", delimiter=",")
validation_data1 = np.loadtxt("validate1.txt", delimiter=",")
validation_data2 = np.loadtxt("validate2.txt", delimiter=",")

x_data = train_data[:,1:]
y_data = train_data[:,0]


x_val1 = validation_data1[:,1:]
x_val2 = validation_data2[:,1:]

y_val1 = validation_data1[:,0]
y_val2 = validation_data2[:,0]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)

X_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
X_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
X_val1 = x_val1.reshape(x_val1.shape[0], 1, img_rows, img_cols)
X_val2 = x_val2.reshape(x_val1.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val1 = X_val1.astype('float32')
X_val2 = X_val2.astype('float32')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_val1 = np_utils.to_categorical(y_val1, nb_classes)
Y_val2 = np_utils.to_categorical(y_val2, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

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

# Callback for model saving:
checkpointer = ModelCheckpoint(filepath="auto_save_weights.hdf5", 
                               verbose=1, save_best_only=True)

t0 = time.time()

# Training
for k in range(1, nb_epoch + 1):
    print("epoch %s/%s:" %(k,nb_epoch))
    X_train_temp = np.copy(X_train) # Copy to not effect the originals
    
    # Add noise on later epochs
    if k > 1:
        for j in range(0, X_train_temp.shape[0]):
            X_train_temp[j,0, :, :] = rand_jitter(X_train_temp[j,0,:,:])

    model.fit(X_train_temp, Y_train, nb_epoch=1, batch_size=batch_size, 
              validation_data=(X_test, Y_test), 
              callbacks=[checkpointer])

t1 = time.time()

test_score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', test_score[0])
print('Test accuracy:', test_score[1])

print("-----------")

val1_score = model.evaluate(X_val1, Y_val1, verbose=0)
print('Val1 score:', val1_score[0])
print('Val1 accuracy:', val1_score[1])

print("-----------")

val2_score = model.evaluate(X_val2, Y_val2, verbose=0)
print('Val2 score:', val2_score[0])
print('Val2 accuracy:', val2_score[1])

print("done took %s sec" % int(t1 - t0))

