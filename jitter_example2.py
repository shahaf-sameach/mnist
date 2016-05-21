import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from scipy.ndimage.interpolation import rotate, shift, zoom
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

# Keras 0.3.2
model = Sequential()
model.add(Convolution2D(32,3,3, border_mode="valid", input_shape=(1,28,28)))
model.add(Activation('relu'))

model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

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

# Parameters
n_epochs = 2

# Training
for k in range(0, n_epochs):
    X_train_temp = np.copy(X_train) # Copy to not effect the originals
    
    # Add noise on later epochs
    if k > 0:
        for j in range(0, X_train_temp.shape[0]):
            X_train_temp[j,0, :, :] = rand_jitter(X_train_temp[j,0,:,:])

    model.fit(X_train_temp, Y_train, nb_epoch=1, batch_size=128, 
              validation_data=(X_test, Y_test), 
              callbacks=[checkpointer])

# Make predictions
predictions = model.predict(X_test, batch_size=128)
print("predictions:", predictions.shape)
# Visualize prediction
print("predictions[1]:", predictions[1])
print("Predicted value: %d" % np.argmax(predictions[1]))

score = model.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])


