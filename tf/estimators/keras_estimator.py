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

    # 1. Compile the model
    model.compile(optimizer='SGD', 
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    # 2. Create an estimator
    model_est = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                    model_dir='./lenet')

    ## Training
    # 3a. Create the training function
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={model.input_names[0]:X['train'].astype(np.float32)},
            y=Y['train'].astype(np.float32),
            batch_size=nb_batches,
            num_epochs=nb_epochs,
            shuffle=True
            )

    # 3b. Train the model
    model_est.train(input_fn=train_input_fn, steps=nb_epochs*nb_batches)

    ## Evaluate
    # 4a. Evaluate the model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={model.input_names[0]:X['test'].astype(np.float32)},
            y=Y['test'].astype(np.float32),
            batch_size=nb_batches,
            num_epochs=nb_epochs,
            shuffle=True
            )

    # 4b. Evaluate the model
    model_eval = model_est.evaluate(input_fn=eval_input_fn, steps=nb_epochs*nb_batches)
    print(model_eval)

    return model_est, model_eval
