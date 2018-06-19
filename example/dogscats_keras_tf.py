from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from dl_utils.data import dogscats
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Lambda

# Model
## lenet_keras for keras API
#from dl_utils.keras.models import lenet_keras
## lenet_tf_keras for tf.keras API
from dl_utils.keras.models import lenet_tf_keras

# Training
## keras training
#from dl_utils.keras.models import train
## tf estimator training
#from dl_utils.tf.estimators import keras_estimator
## tf training
from dl_utils.tf.models import train

# Log level
tf.logging.set_verbosity(tf.logging.DEBUG)

# Constant values
NB_EPOCHS = 10
NB_BATCHES = 128
NB_CLASSES = 2
NB_ROWS = 224
NB_COLS = 224
NB_CHANNELS = 3


def add_custom_layers(x):
    img_path = tf.read_file(tf.squeeze(tf.cast(x, tf.string)))
    img_u8 = tf.image.decode_jpeg(img_path, channels=3)
    img = tf.image.convert_image_dtype(img_u8, dtype=tf.float32)
    img_cropped = tf.image.resize_image_with_crop_or_pad(img, 224, 224)
    #img_cropped = tf.expand_dims(img_cropped, 0)
    return img_cropped

def resize_images(x):
    return tf.map_fn(add_custom_layers, x, dtype=tf.float32)

# Our application logic will be added here
def main():
    
    x_ph = tf.placeholder(tf.string, shape=None)
    y_ph = tf.placeholder(tf.float32, shape=(None,NB_CLASSES))

    # 1. Loading the data
    print('Loading data...')
    file_path = '/mnt/data/Personal/coursera/DL/data/dogscats/sample'
    X, Y = dogscats.load_data(file_path)

    # 2. Create the model
    # 2a. Lambda layer
    in_data = Input(batch_shape=(None,1), dtype=tf.string, name='input')
    img_cropped = Lambda(resize_images, output_shape=(None, 224, 224, 3))(in_data)
    
    # 2b. LeNet model
    model1 = lenet_tf_keras.model(nb_classes=NB_CLASSES, nb_rows=NB_ROWS, nb_cols=NB_COLS, nb_channels=NB_CHANNELS) 
    
    # 2c. merged model
    model = Model(inputs=in_data, outputs=model1(img_cropped))
    model1.summary()
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
    train.run(model, X, Y_oh, x_ph=x_ph, y_ph=y_ph, model_path='./saved_model/lenet_dogscats',nb_epochs=NB_EPOCHS, nb_batches=NB_BATCHES, nb_rows=NB_ROWS, 
            nb_cols=NB_COLS, nb_channels=NB_CHANNELS, nb_classes=NB_CLASSES)

if __name__ == "__main__":
  #tf.app.run()
  main()
