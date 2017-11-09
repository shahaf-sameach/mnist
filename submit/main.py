from __future__ import print_function
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from jitter import rand_jitter
from time import time
from network import Network
# from callbacks import ScoreHistory

def write_to_file(predictions, filename='predictions.txt'):
  with open(filename, 'w') as predictions_file:
    for predict in predictions:
      predictions_file.write("%s\n" % np.argmax(predict))

  print("wrote predictions to %s" % filename)


def load_data_from_file(filename="train.txt"):
  data = np.loadtxt(filename, delimiter=",", converters = {0: lambda s: 0 if s == '?' else s})
  # seperate to X and Y
  x_data = data[:,1:]
  y_data = data[:,0]
  X_data = x_data.reshape(x_data.shape[0], 1, 28, 28)
  X_data = X_data.astype('float32')
  Y_data = np_utils.to_categorical(y_data, 10)

  return X_data, Y_data


if __name__ == '__main__':
  np.random.seed(1337)  # for reproducibility

  print("Please choose run mode: train[t] or predict[p], press any other key to abort...")
  choise = raw_input()
  
  nnet = Network()
  weight_file = "weights.hdf5"

  if choise == 't':
    print("Loading data sets...")
    x_data, y_data = load_data_from_file("train.txt")
    
    # split to train and test
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.1)

    X_val1, Y_val1 = load_data_from_file("validate1.txt")
    X_val2, Y_val2 = load_data_from_file("validate2.txt")

    # train the model
    print("Training model...")
    nnet.train([X_train, Y_train], [X_test, Y_test], nb_epoch=100, weight_file=weight_file)
    # load the best weights
    nnet.load_weights(weight_file)

    
    print("Evaluating model...")
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

  elif choise == 'p':
    # load the best weights
    nnet.load_weights(weight_file)

    # Make predictions
    # Load data from file
    print("Loading test set...")
    X_test, Y_test = load_data_from_file("test.txt")

    # predict using the model
    print("predicting...")
    predictions = nnet.predict(X_test, batch_size=128)

    # write to file
    write_to_file(predictions)
    
  else:
    pass

  print("BYE")

