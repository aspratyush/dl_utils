from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import math
import numpy as np
from sklearn.utils import shuffle

def run(model, X, Y, x_ph=None, y_ph=None, model_path='/tmp', optimizer=None, 
        nb_epochs=30, nb_batches=128, nb_rows=28, nb_cols=28, nb_channels=1, nb_classes=10):

    # define the placeholders
    if (x_ph is None):
        x_ph = tf.placeholder(tf.float32, shape=(None, nb_rows, nb_cols, nb_channels))
    if (y_ph is None):
        y_ph = tf.placeholder(tf.float32, shape=(None, nb_classes))
    # get the prediction tensor
    y_pred = model(x_ph)

    # 1. loss function
    with tf.name_scope('loss'):
        cross_entropy = tf.losses.softmax_cross_entropy(y_ph, y_pred)
        cross_entropy = tf.reduce_mean(cross_entropy)
    
    # 2. optimizer
    with tf.name_scope('model_optimizer'):
        if (optimizer is None):
            train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
        else:
            # TODO check!!!
            train_step = optimizer.minimize(cross_entropy)
    
    # 3. accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_ph,1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
    
    # 4. run the training
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # intialize variables
        sess.run(tf.global_variables_initializer())
        
        # steps
        nb_steps = math.floor(np.asarray(X['train']).shape[0]/nb_batches)
        
        X_train = X['train']
        Y_train = Y['train']
        if (Y_train.shape[1] != nb_classes):
            Y_train = tf.keras.utils.to_categorical(Y_train, nb_classes)

        # iterate over the epochs
        print('Training...')
        for epoch in range(nb_epochs):

            # shuffle
            X_train, Y_train = shuffle(X_train, Y_train)
        
            # Take the steps
            for i in range(nb_steps):
            
                # Extract the batch
                x_batch = X_train[i*nb_batches:(i+1)*nb_batches]
                y_batch = Y_train[i*nb_batches:(i+1)*nb_batches]

                # run the optimizer
                sess.run(train_step, feed_dict={x_ph:x_batch, y_ph:y_batch})
            
            # 5. Validation
            # print accuracy every 10 epochs 
            if (epoch % 2 == 0):
                val_accuracy = sess.run(accuracy, feed_dict={x_ph: X['valid'], y_ph: Y['valid']})
                val_loss = sess.run(cross_entropy, feed_dict={x_ph: X['valid'], y_ph: Y['valid']})
                print('Epoch: ', epoch, ', Val accuracy: ', val_accuracy, ', Val loss : ', val_loss)

        # Save the model
        print('Saving the model...')
        saver.save(sess, model_path)
