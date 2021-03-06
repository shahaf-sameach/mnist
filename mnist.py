'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import time
# from keras.utils.visualize_util import plot

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

data = np.loadtxt("train.txt", delimiter=",")

precent = int(data.shape[0] * 0.1)
test_data, train_data = data[:precent,:], data[precent:,:]

x_train = train_data[:,1:]
x_test = test_data[:,1:]

std_dev = np.std(x_train, axis=0)

y_train = train_data[:,0]
y_test = test_data[:,0]

X_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
X_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

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


t0 = time.time()

# Training
for k in range(0, nb_epoch):
    X_train_temp = np.copy(x_train) # Copy to not effect the originals
    
    # Add noise on later epochs
    if k > 0:
        for j in range(0, X_train_temp.shape[0]):
            X_train_temp[j,0, :, :] = rand_jitter(X_train_temp[j,0,:,:])

    model.fit(X_train_temp, Y_train, nb_epoch=1, batch_size=batch_size, 
              validation_data=(X_test, Y_test), 
              callbacks=[checkpointer])

t1 = time.time()

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
print("done took %s sec" % int(t1 - t0))

