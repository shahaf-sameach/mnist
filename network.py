from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ModelCheckpoint
from jitter import rand_jitter
import numpy as np


class Network(object):

  def __init__(self, nb_filters=32, nb_conv=3, nb_pool=2):
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, 28, 28)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Convolution2D(nb_filters/2, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(500))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    self.model = model

  def get_model(self):
    return self.model

  def train(self, train_data, validation_data, nb_epoch=50, batch_size=128, weight_file="weights.hdf5"):
    # Callback for model saving:
    checkpointer = ModelCheckpoint(filepath=weight_file, 
                               verbose=1, save_best_only=True)

    X_train, Y_train = train_data[0], train_data[1]
    X_test, Y_test = validation_data[0], validation_data[1]

    # Training
    for k in range(1, nb_epoch + 1):
        print("epoch %s/%s:" %(k,nb_epoch))
        X_train_temp = np.copy(X_train) # Copy to not effect the originals
        
        # Add noise on later epochs
        if k > 1:
            for j in range(0, X_train_temp.shape[0]):
                X_train_temp[j,0, :, :] = rand_jitter(X_train_temp[j,0,:,:])

        self.model.fit(X_train_temp, Y_train, nb_epoch=1, batch_size=batch_size, 
                  validation_data=(X_test, Y_test), 
                  callbacks=[checkpointer])

  def load_weights(self, weight_file):
    self.model.load_weights(weight_file)

  def evaluate(self, X_data, Y_data):
    score = self.model.evaluate(X_data, Y_data, verbose=0)
    return score

  def predict(self, X_test, batch_size=128):
    predictions = self.model.predict(X_test, batch_size=batch_size)
    return predictions


if __name__ == '__main__':
  a = Network()
  model = a.get_model()

