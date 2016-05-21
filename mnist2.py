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
import random
# from keras.utils.visualize_util import plot


def rand_jitter(temp):
	basicTemp = temp
	moreTemp = temp
	#if np.random.random() > .7:
	temp = basicTemp
	moreTemp[np.random.randint(0,28,1), :] = 0
	temp += moreTemp
	if np.random.random() > .7:
		temp = basicTemp
		moreTemp[:, np.random.randint(0,28,1)] = 0
		temp += moreTemp
	if np.random.random() > .7:
		temp = basicTemp
		moreTemp = shift(moreTemp, shift=(np.random.randint(-3,3,2)))
		temp += moreTemp
	if np.random.random() > .7:
		temp = basicTemp
		moreTemp = rotate(moreTemp, angle = np.random.randint(-15,15,1), reshape=False)
		temp += moreTemp
	return temp


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

x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


startX_train = x_train

#add gaussian noise
X_trainMore = startX_train
rand_jitter(X_trainMore)
x_train += X_trainMore

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

model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(x_test, y_test))

t1 = time.time()

score = model.evaluate(x_test, y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save_weights("weights_%s_%s.hdf5" %(score[0], score[1]))

print("done took %s sec" % int(t1 - t0))
