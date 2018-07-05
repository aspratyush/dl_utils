from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
import numpy as np
import tensorflow as tf


def run(model, X, Y, optimizer=None, nb_epochs=30, nb_batches=128):
    """
    Run the estimator
    """
    if optimizer is None:
        optimizer = tf.keras.estimators.SGD(
                lr=0.0009, decay=1e-5, momentum=0.9, nesterov=True)

    # 1. Compile the model
    model.compile(
            optimizer=optimizer, loss='categorical_crossentropy',
            metrics=['accuracy'])

    # 2. Create an estimator
    model_est = tf.keras.estimator.model_to_estimator(
            keras_model=model, model_dir='./lenet')

    # Training
    # 3a. Create the training function
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={model.input_names[0]: X['train'].astype(np.float32)},
            y=Y['train'].astype(np.float32),
            batch_size=nb_batches,
            num_epochs=nb_epochs,
            shuffle=True
            )

    # 3b. Train the model
    model_est.train(input_fn=train_input_fn, steps=nb_epochs*nb_batches)

    # Evaluate
    # 4a. Evaluate the model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={model.input_names[0]: X['test'].astype(np.float32)},
            y=Y['test'].astype(np.float32),
            batch_size=nb_batches,
            num_epochs=nb_epochs,
            shuffle=True
            )

    # 4b. Evaluate the model
    model_eval = model_est.evaluate(input_fn=eval_input_fn)
    print(model_eval)

    return model_est, model_eval


def run_from_generator(
        model, input_func=None, input_func_dict=None,
        eval_func_dict=None, nb_epochs=10, optimizer=None, model_dir=None):
    """
    Overloaded function to create an estimator using tf.data.Dataset
    :param model : uncompiled keras model
    :param input_fn : input function providing tf.data.Dataset to the estimator
    :param input_fn_dict : dictionary containing input params for input_fn
    :param eval_fn_dict : dictionary containing params for eval input_fn
    :param model_dir : directory to store the trained model
    """

    # 1. Create optimizer and compile model if optimizer is None
    if (optimizer is None):
        optimizer = tf.keras.optimizers.SGD(
                lr=1e-3, decay=1e-5, momentum=0.9, nesterov=True)

    # 2. compile the model
    model.compile(
            optimizer=optimizer, loss='categorical_crossentropy',
            metrics=['accuracy'])

    # 3. create estimator
    dir_path = os.path.join(os.getcwd(), model_dir)
    print("Model path chosen : ", dir_path)
    if (not os.path.exists(dir_path)):
        os.mkdir(dir_path)

    print("Creating estimator...")
    est = tf.keras.estimator.model_to_estimator(
            keras_model=model, model_dir=dir_path)

    # 4. Train and Evaluate the model
    print("Training...")

    # training spec
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_func(input_func_dict),
            max_steps=500)

    # evaluation spec
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_func(eval_func_dict))

    # Run the training
    model_est = tf.estimator.train_and_evaluate(est, train_spec, eval_spec)
    #est.train(input_fn=lambda: input_func(input_func_dict),
    #        steps=None)
    #
    #est.evalute(input_fn=lambda: input_func(eval_func_dict))

    return est
