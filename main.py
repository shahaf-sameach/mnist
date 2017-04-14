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
from callbacks import ScoreHistory
# from keras.utils.visualize_util import plot

np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_classes = 10
nb_epoch = 50

weight_file = "weights7.hdf5"

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# load data from files
train_data = np.loadtxt("train.txt", delimiter=",")
validation_data1 = np.loadtxt("validate1.txt", delimiter=",")
validation_data2 = np.loadtxt("validate2.txt", delimiter=",")

# seperate to X and Y
x_data = train_data[:,1:]
y_data = train_data[:,0]

x_val1 = validation_data1[:,1:]
x_val2 = validation_data2[:,1:]

y_val1 = validation_data1[:,0]
y_val2 = validation_data2[:,0]

# split to train and test
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)

# reshape data
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

# load the model
nnet = Network()
model = nnet.model

print("32x16 random noise")
t0 = time()
nnet.train([X_train, Y_train], [X_test, Y_test], weight_file=weight_file)
# Training
# for k in range(1, nb_epoch + 1):
#     print("epoch %s/%s:" %(k,nb_epoch))
#     X_train_temp = np.copy(X_train) # Copy to not effect the originals
    
#     # Add noise on later epochs
#     if k > 1:
#         for j in range(0, X_train_temp.shape[0]):
#             X_train_temp[j,0, :, :] = rand_jitter(X_train_temp[j,0,:,:])

#     model.fit(X_train_temp, Y_train, nb_epoch=1, batch_size=batch_size, 
#               validation_data=(X_test, Y_test), 
#               callbacks=[checkpointer])

t1 = time()

# load the best weights
nnet.load_weights(weight_file)

# evaluate model based on the test set (split from the train set)
test_score = nnet.evaluate(X_test, Y_test)
print('Test score:', test_score[0])
print('Test accuracy:', test_score[1])

print("-----------")

# evaluate model based on the validate1 set
val1_score = nnet.evaluate(X_val1, Y_val1)
print('Val1 score:', val1_score[0])
print('Val1 accuracy:', val1_score[1])

print("-----------")

# evaluate model based on the validate2 set
val2_score = nnet.evaluate(X_val2, Y_val2)
print('Val2 score:', val2_score[0])
print('Val2 accuracy:', val2_score[1])

print("done took %s sec" % int(t1 - t0))
