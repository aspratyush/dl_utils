from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf


def create(data_dir='/tmp'):
    """
    create summary writer
    """
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(data_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(data_dir + '/test')

    return [merged, train_writer, test_writer]


def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
