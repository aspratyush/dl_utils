import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from dl_utils.tf.plot_weights import plot_weights

# CUDA GPU
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'


def get_stats(X,Y):
    print("X : shape : (%d,%d)" % (X.shape), end='')
    print(",min : %f, max : %f" % (np.min(X), np.max(X)))
    print("Y : shape : (%d,%d)" % (Y.shape), end='')
    print(", min : %f, max : %f" % (np.min(Y), np.max(Y)))


def load_data(one_hot=False, nb_classes=10):
    from tensorflow.examples.tutorials.mnist import input_data

    # load data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=one_hot)
    x_train, y_train = mnist.train.images, mnist.train.labels
    x_test, y_test = mnist.test.images, mnist.test.labels
    x_validation, y_validation = mnist.validation.images, mnist.validation.labels

    if not(one_hot):
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=nb_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=nb_classes)
        y_validation = tf.keras.utils.to_categorical(y_validation, num_classes=nb_classes)

    # print stats
    print("train : ")
    get_stats(x_train, y_train)
    print("test : ")
    get_stats(x_test, y_test)
    print("validation : ")
    get_stats(x_validation, y_validation)

    return mnist, x_train, y_train, x_test, y_test, x_validation, y_validation


def build_model(use_softmax=False):
    model = Sequential()
    model.add(Dense(10, input_shape=(None, 784), activation='relu'))
    #model.add(Dense(84, activation='relu'))
    #model.add(Dense(10))

    # softmax
    if use_softmax:
        model.add(Activation('softmax'))

    return model


def main():
    print('In main...')
    # 1. load the data
    data, x_train, y_train, x_test, y_test, x_validation, y_validation = load_data()

    # 2. create model
    model = build_model()

    #3. get logits
    x = tf.placeholder(dtype=tf.float32, shape=(None, 784))
    y = tf.placeholder(dtype=tf.float32, shape=(None, 10))
    logits = model(x)
    model.summary()

    # 4. get loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
    cost = tf.reduce_sum(cross_entropy)

    # 5. Optimization
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

    # 6. Performance checks
    y_pred = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 7. session run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        nb_epochs = 100
        nb_batches = 256
        for epoch in range(nb_epochs):
            avg_cost = 0
            # shuffle
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(0, int(x_train.shape[0]/nb_batches)):
                start = j*nb_batches
                end = (j+1)*nb_batches
                if end > x_train.shape[0]:
                    end = x_train.shape[0]
                x_batch, y_batch = x_train[start:end,:], y_train[start:end,:]
                # run optimization on this batch
                _, c = sess.run([optimizer,cost], feed_dict={x:x_batch, y:y_batch})
                avg_cost += c/nb_batches

            # Display results
            if epoch % 10 == 0:
                acc = sess.run(accuracy, feed_dict={x:x_validation, y:y_validation})
                print("Epoch:", '%04d' % (epoch+1),
                        "cost={:.9f}".format(avg_cost),
                        "accuracy=", acc)
                layer_weights = model.layers[0].get_weights()[0]
                plot_weights(layer_weights, (28,28), idx=epoch)

        print("Optimization finished...")
        plt.show()
        ## 8. Test accuracy
        #acc = sess.run(accuracy, feed_dict={x:x_test, y:y_test})
        #print("Test accuracy = ", acc)

if __name__ == '__main__' :
    main()
