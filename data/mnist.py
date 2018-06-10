"""
forked from : https://github.com/tensorflow/cleverhans/blob/master/cleverhans/utils_mnist.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import warnings
from tensorflow.examples.tutorials.mnist import input_data
from . import data_info

def load_data(data_dir='/tmp', one_hot=False):
    """
    Load and preprocess MNIST dataset
    :param datadir: path to folder where data should be stored
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: tuple of four arrays containing training data, training labels,
             testing data and testing labels.
    """
    # load the data
    mnist = input_data.read_data_sets(data_dir, one_hot=one_hot, reshape=False)

    X = {}
    Y = {}
    # split into train - test - validation
    X['train'] = mnist.train.images
    Y['train'] = mnist.train.labels
    X['test'] = mnist.test.images
    Y['test'] = mnist.test.labels
    X['val'] = mnist.validation.images
    Y['val'] = mnist.validation.labels

    # print shapes
    data_info.get_shapes(X, Y)

    return X, Y
