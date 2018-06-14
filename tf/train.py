from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf

def run(model, X, Y, optimizer=None, nb_epochs=30, nb_batches=128):
    
    # TODO check
    y_pred = model(x)

    # 1. loss function
    with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
    
    # 2. optimizer
    with tf.name_scope('model_optimizer'):
        if (optimizer is None):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        else:
            # TODO check!!!
            train_step = optimizer(1e-3).minimize(cross_entropy)
    
    # 3. accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), Y['train'])
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    
    # 4. run the training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # TODO check
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if (i % 100 == 0):
                train_accuracy = accuracy.eval(
                        feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            
        print('test accuracy %g' % accuracy.eval(
            feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
