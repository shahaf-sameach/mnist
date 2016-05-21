import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from scipy.ndimage.interpolation import rotate, shift, zoom
from keras.datasets import mnist

# Load data
train = pd.read_csv('train.csv', dtype=np.float32)
X_test = pd.read_csv('test.csv', dtype=np.float32).values

np.random.seed(182)

from sklearn.cross_validation import train_test_split
    
y_train = pd.get_dummies(train.loc[:, "label"])
print(y_train.head())

# Get numpy version
y_train = y_train.values.astype(np.int32)
X_train = train.drop("label", axis = 1).values

X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                      y_train, 
                                                      test_size=0.1)

X_train = X_train.reshape((X_train.shape[0],1,28,28))/255
X_valid = X_valid.reshape((X_valid.shape[0],1,28,28))/255
X_test = X_test.reshape((X_test.shape[0],1,28,28))/255

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

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

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

# Copy to not effect the original
ind = np.random.randint(len(X_train))
test_image = lambda : np.copy(X_train[ind,0,:,:])

# Jitter examples
# plt.figure()
# f, ax = plt.subplots(2, 2)
# for k in range(2):
#     for j in range(2):
#         ax[k,j].imshow(rand_jitter(test_image()))

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

    model.fit(X_train_temp, y_train, nb_epoch=1, batch_size=128, 
              validation_data=(X_valid, y_valid), show_accuracy=True, verbose=1, 
              callbacks=[checkpointer])

# Make predictions
predictions = model.predict(X_test, batch_size=128)


# Visualize prediction
print("Predicted value: %d" % np.argmax(predictions[1]))
plt.imshow(X_test[1,0,:,:])

