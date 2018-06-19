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
    tf.summary.FileWriterCache.clear()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(data_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(data_dir + '/test')

    return merged, train_writer, test_writer


def close(train_writer, test_writer):
    """
    Closing the summaries file
    """
    train_writer.close()
    test_writer.close()


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


def scalar_summaries(x, name):
    """
    Attach scalar summary to a tensor with name=name
    """
    tf.summary.scalar(name, x)


def image_summaries(x, name):
    """
    Attach image summary to a image tensor with name=name
    """
    tf.summary.image(name, x)
