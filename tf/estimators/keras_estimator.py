from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD

def run(model, X, Y, optimizer=None, nb_epochs=30, nb_batches=128):
    """
    Run the estimator
    """
    assert (model, not None), "model cannot be None!"

    if optimizer is None:
        optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # Compile the model
    model.compile(optimizer='SGD', 
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    # Create an estimator
    model_est = tf.keras.estimator.model_to_estimator(keras_model=model)

    # Create the training function
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={model.input_names[0]:X['train'].astype(np.float32)},
            y=Y['train'].astype(np.float32),
            num_epochs=nb_epochs,
            shuffle=True
            )

    # Train the model
    model_est.train(input_fn=train_input_fn, steps=2000)

    return model_est
