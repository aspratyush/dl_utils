from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from dl_utils.data import mnist

# lenet_keras for keras API
from dl_utils.keras.models import lenet_keras
# lenet_tf_keras for tf.keras API
from dl_utils.keras.models import lenet_tf_keras

#from dl_utils.keras.models import train
from dl_utils.tf.estimators import keras_estimator
from dl_utils.tf.models import train
# Log level
tf.logging.set_verbosity(tf.logging.DEBUG)

# Our application logic will be added here
def main():

    # Constant values
    NB_EPOCHS = 10
    NB_BATCHES = 128
    NB_CLASSES = 10
    NB_ROWS = 28
    NB_COLS = 28
    NB_CHANNELS = 1

    # 1. Loading the data
    print('Loading data...')
    X, Y = mnist.load_data()

    # 2. Create the model
    model = lenet_tf_keras.model(nb_classes=NB_CLASSES) 
    model.summary()

    #3. get one-hot encoded vectors for the labels
    Y_oh = {}
    for key in Y.keys():
        Y_oh[key] = tf.keras.utils.to_categorical(Y[key], NB_CLASSES)
    
    # 4. train
    ## a. keras
    #train.run(model, X, Y_oh, nb_epochs=5)

    ## b. estimator
    #keras_estimator.run(model, X, Y_oh, nb_epochs=30, nb_batches=128)

    ## c. tf
    train.run(model, X, Y_oh, model_path='./saved_model/lenet',nb_epochs=NB_EPOCHS, nb_batches=NB_BATCHES, nb_rows=NB_ROWS, 
            nb_cols=NB_COLS, nb_channels=NB_CHANNELS, nb_classes=NB_CLASSES)

if __name__ == "__main__":
  #tf.app.run()
  main()
